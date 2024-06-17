from __future__ import annotations

from bisect import bisect_left
from bisect import bisect_right

import numpy as np
import pandas as pd

from scipy.stats import chi2

class SurvInfo:
    def __init__(
        self,
        time: int,
        events_count: int,
        censored_count: int,
        at_risk_count: int,
        probability: float,
    ) -> None:
        self.time: int = time
        self.events_count: int = events_count
        self.censored_count: int = censored_count
        self.at_risk_count: int = at_risk_count
        self.probability: float = probability

from bisect import bisect_left
 
def binary_search(arr, target):
    index = bisect_left(arr, target)
    if index < len(arr) and arr[index] == target:
        return index
    else:
        return (-index - 1)
    
class KaplanMeierEstimator:

    def __init__(
        self,
        surv_info_list: list[SurvInfo] = []
    ) -> None:
        self.surv_info_list: list[SurvInfo] = surv_info_list

    def fit(self,survival_time: np.ndarray, survival_status:  np.ndarray) -> KaplanMeierEstimator:

        info_list: list[SurvInfo] = []
        if survival_time.shape[0] == 0:
            return self
        for status, time in zip(survival_status, survival_time):
            is_censored = int((status == 0))
            is_event = int((status == 1))
            info_list.append(SurvInfo(time = time, events_count = is_event, censored_count = is_censored, at_risk_count = 0, probability = 0))

        # sort surv_info_list by survival_time
        info_list.sort(key=lambda x: x.time, reverse=False)

  
        at_risk_count = survival_time.shape[0]
        grouped_data = {}
        time_point_prev = info_list[0]
        for time_point in info_list:
            if time_point.time != time_point_prev.time:
                grouped_data[time_point_prev.time].at_risk_count = at_risk_count
                at_risk_count -= grouped_data[time_point_prev.time].events_count
                at_risk_count -= grouped_data[time_point_prev.time].censored_count
                time_point_prev = time_point

            if time_point.time in grouped_data:
                grouped_data[time_point.time].events_count += time_point.events_count
                grouped_data[time_point.time].censored_count += time_point.censored_count
            else:
                grouped_data[time_point.time] = SurvInfo(
                    time_point.time, time_point.events_count, time_point.censored_count, time_point.at_risk_count, time_point.probability
        )
                
        grouped_data[time_point.time].at_risk_count = at_risk_count


        self.surv_info_list = [info for info in grouped_data.values()]

        self.calculate_probabilities(self.surv_info_list)

        return self

            
    def calculate_probabilities(self, surv_info_list: list[SurvInfo]) -> list[SurvInfo]:
        for i in range(len(surv_info_list)):
            surv_info = surv_info_list[i]
            if surv_info.at_risk_count == 0:
                surv_info.probability = 0
            else:
                surv_info.probability = (surv_info.at_risk_count - surv_info.events_count) / surv_info.at_risk_count
            if i > 0:
                surv_info.probability *= surv_info_list[i - 1].probability

    
    @staticmethod
    def average(estimators: list[KaplanMeierEstimator]) -> KaplanMeierEstimator:

        unique_times = set()
        for estimator in estimators:
            unique_times.update([info.time for info in estimator.surv_info_list])

        unique_times = sorted(list(unique_times))


        probabilities = dict()
        for time in unique_times:
            p = 0
            for estimator in estimators:
                p+= estimator.get_probability_at(time)
            probabilities[time] = p/len(estimators)

        
        avg_estimator = KaplanMeierEstimator()
        avg_estimator.surv_info_list = [SurvInfo(time, 0, 0, 0, probabilities[time]) for time in unique_times]

        return avg_estimator
    
    def get_probability_at(self, time: int) -> float:
        index = binary_search([info.time for info in self.surv_info_list], time)
        
        if index >= 0:
            return self.surv_info_list[index].probability
        
        index = ~index

        if index == len(self.surv_info_list):
            return self.surv_info_list[index - 1].probability
        
        if index == 0:
            return 1
        
        return self.surv_info_list[index - 1].probability
        
    

    def get_events_count_at(self, time: int) -> int:
        index = binary_search([info.time for info in self.surv_info_list], time)
        if index >=0:
            return self.surv_info_list[index].events_count
        
        return 0
    
    
    def get_at_risk_count_at(self, time: int) -> int:
        index = binary_search([info.time for info in self.surv_info_list], time)
        if index >=0:
            return self.surv_info_list[index].at_risk_count
        
        index = ~index

        n = len(self.surv_info_list)
        if index == n:
            return self.surv_info_list[n - 1].at_risk_count
        
        return self.surv_info_list[index].at_risk_count
    
    def get_times(self) -> list[int]:
        return [info.time for info in self.surv_info_list]
    
    def get_probabilities(self) -> list[float]:
        return [info.probability for info in self.surv_info_list]

    def reverse(self) -> KaplanMeierEstimator:
        revKm = KaplanMeierEstimator(surv_info_list=[])

        for info in self.surv_info_list:
            revKm.surv_info_list.append(SurvInfo(time=info.time, events_count=info.censored_count, censored_count=info.events_count, at_risk_count=info.at_risk_count, probability=0)) 

        revKm.calculate_probabilities(revKm.surv_info_list)

        return revKm
    

    @staticmethod
    def compare_estimators(kme1: KaplanMeierEstimator, kme2: KaplanMeierEstimator)-> dict(str, float):
        
        results = dict()

        if (len(kme1.get_times()) == 0) or (len(kme2.get_times()) == 0):
            results["stats"] = 0
            results["p_value"] = 0
            return results
            

        times = set(kme1.get_times())
        times.update(kme2.get_times())

        x = 0
        y = 0

        for time in times:
            m1 = kme1.get_events_count_at(time)
            n1 = kme1.get_at_risk_count_at(time)

            m2 = kme2.get_events_count_at(time)
            n2 = kme2.get_at_risk_count_at(time)


            e2 = (n2 / (n1 + n2)) * (m1 + m2)
            n = n1 + n2

            

            x += m2 - e2
            if (n * n * (n - 1)) == 0:
                y+= 0
            else:
                y += (n1 * n2 * (m1 + m2) * (n - m1 - m2)) / (n * n * (n - 1))

        results["stats"] = (x * x) / y
        results["p_value"] = 1 - chi2.cdf(results["stats"], 1)

        return results
    

    