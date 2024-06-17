"""
Contains survival ruleset class.
"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from decision_rules.core.coverage import Coverage
from decision_rules.core.coverage import CoverageInfoDict
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.survival.rule import SurvivalConclusion
from decision_rules.survival.rule import SurvivalRule
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator

class SurvivalRuleSet(AbstractRuleSet):
    """Survival ruleset allowing to perform prediction on data
    """

    def __init__(
        self,
        rules: List[SurvivalRule],
    ) -> None:
        """
        Args:
            rules (List[SurvivalRule]):
        """
        self.rules: List[SurvivalRule]
        self.survival_time_attr_name: Optional[str] = None
        self.survival_status_attr_name: Optional[str] = None
        super().__init__(rules)
    
    def set_survival_time_attr_name(self, name: str):
        self.survival_time_attr_name = name
    
    def set_survival_status_attr_name(self, name: str):
        self.survival_status_attr_name = name

    def _calculate_P_N(self, y_uniques: np.ndarray, y_values_count: np.ndarray):  # pylint: disable=invalid-name
        return
    
    def update_using_coverages(
        self,
        coverages_info: Dict[str, CoverageInfoDict],
        colum_names: List[str],
        y_median: float,
        measure: Callable[[Coverage], float]
    ):
        raise NotImplementedError("You should call update method instead. This method is not implemented for survival ruleset, beacause information only about coverage are not sufficient to determine default conclusion.")   

    def update(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        measure: Callable[[Coverage], float]
    ):
        self.default_conclusion = SurvivalConclusion(
            estimator=KaplanMeierEstimator().fit(X_train[self.survival_time_attr_name].to_numpy(), y_train.to_numpy()),
            column_name=self.rules[0].conclusion.column_name
        )

        if len(self.rules) == 0:
            raise ValueError(
                '"update" cannot be called on empty ruleset with no rules.'
            )
        #X_train =X_train.drop(self.survival_time_attr_name, axis=1)
        self.column_names = X_train.columns.tolist()
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        self.calculate_rules_coverages(X_train, y_train)

    
    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Args:
            X (pd.DataFrame)
        Returns:
            List[KapleinMaier]: prediction
        """

        X = (X[self.column_names]).to_numpy()
        return self._perform_prediction(X)

    def _perform_prediction(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        # Based on article: Wróbel et al. Learning rule sets from survival data BMC Bioinformatics (2017) 18:285 Page 5 of 13
        # The learned rule set can be applied for an estimation of the survival function of new observations based on the values taken by their covariates. 
        # The estimation is per formed by rules covering given observation. If observation is not covered by any of the rules then it has assigned the default survival estimate computed on the entire train ing set. 
        # Otherwise, final survival estimate is calculated as an average of survival estimates of all rules covering the observation

        prediction = []
        for i in range(X.shape[0]):
            km = self._predict_for_example(X[i].reshape(1,-1), self.rules)
            prediction_for_exmple = dict()
            prediction_for_exmple["time"] = km.get_times()
            prediction_for_exmple["probability"] = km.get_probabilities()
            prediction.append(prediction_for_exmple)

        
        return prediction
            
    def _predict_for_example(self, example: np.ndarray, rules: List[SurvivalRule]) -> float:
        matching_rules = []
        for rule in rules:
            if (rule.premise.covered_mask(example)[0]):
                matching_rules.append(rule)
        
        if len(matching_rules) == 0:
            kaplan = self.default_conclusion.value
        else:
            survfits = []
            for rule in matching_rules:
                survfits.append(rule.conclusion.value)

            kaplan = KaplanMeierEstimator.average(survfits)
            
        return kaplan


    def _prepare_prediction_array(self, X: np.ndarray) -> np.ndarray:
        pass

    def calculate_rules_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[Dict[str, str, float]]:
        raise NotImplementedError()
    
    def calculate_condition_importances(self, X_train: pd.DataFrame, y_train: pd.Series, measure: Callable[[Coverage], float]) -> Dict[str, float]:
        raise NotImplementedError("This function is not implemented yet")
    
    def calculate_attribute_importances(self, condition_importances: Dict[str, float]) -> Dict[str, float]:
        raise NotImplementedError("This function is not implemented yet")
    

    def integrated_bier_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        X_numpy = X.to_numpy()
        survival_times = X[self.survival_time_attr_name].to_numpy()
        survival_status = y.to_numpy()
        censoring_KM = self.default_conclusion.value.reverse()
        info_list: list[IBSInfo] = []
        for i in range(X.shape[0]):
            km = self._predict_for_example(X_numpy[i].reshape(1,-1), self.rules)
            is_censored = (survival_status[i] == 0)
            time = survival_times[i]
            info = IBSInfo(time,is_censored,km)
            info_list.append(info)

        # sort info list by time
        sorted_info = sorted(info_list, key=lambda x: x.time, reverse=False)
        info_size = len(sorted_info)
        brier_score: list[float]= []

        for i in range(len(sorted_info)):
            bt = sorted_info[i].time
            if bt == 1397:
                print("1397")
            if (i > 0) and bt == sorted_info[i-1].time:
                brier_score.append(sorted_info[i-1].time)
            else:
                brier_sum = 0
                for si in sorted_info:
                    if si.time <= bt and si.is_censored == False:
                        g = censoring_KM.get_probability_at(si.time)
                        if g>0:
                            p = si.estimator.get_probability_at(bt)
                            brier_sum += (p*p)/g
                    elif si.time > bt:
                        g = censoring_KM.get_probability_at(bt)
                        if g>0:
                            p = 1 - si.estimator.get_probability_at(bt)
                            brier_sum += (p*p)/g
                
                brier_score.append(brier_sum/info_size)

        diffs: list[float] = []
        diffs.append(sorted_info[0].time)
        for i in range(1, info_size):
            diffs.append(sorted_info[i].time - sorted_info[i - 1].time)


        sum = 0

        for i in range(info_size):
            sum += diffs[i] * brier_score[i]

        score = sum / sorted_info[info_size - 1].time

        return score
        


class IBSInfo:
    def __init__(
        self,
        time: float,
        is_censored: bool,
        estimator: KaplanMeierEstimator,
    ) -> None:
        self.time = time
        self.is_censored = is_censored
        self.estimator = estimator