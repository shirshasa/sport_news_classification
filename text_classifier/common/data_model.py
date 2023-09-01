from typing import List, Dict, Union


Probability = float
Category = str
Prediction = Dict[str, Union[Category, Probability]]
SerializedPredictions = List[dict]
