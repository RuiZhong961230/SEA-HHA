# SEA-HHA
Surrogate Ensemble-Assisted hyper-heuristic algorithm for Expensive Optimization Problems

## Abstract
This paper proposes a novel surrogate ensemble-assisted hyper-heuristic algorithm (SEA-HHA) to solve expensive optimization problems (EOPs). A representative HHA consists of two parts: the low-level and the high-level components. In the low-level component, we regard the surrogate-assisted technique as a type of search strategy and design the four search strategy archives: exploration strategy archive, exploitation strategy archive, surrogate-assisted estimation archive, and mutation strategy archive as low-level heuristics (LLHs), each archive contains one or more search strategies. Once the surrogate-assisted estimation archive is activated to generate the offspring individual, SEA-HHA first selects the dataset for model construction from three principles: All Data, Recent Data, and Neighbor, which correspond to the global and the local surrogate model, respectively. Then, the dataset is randomly divided into training and validation data, and the most accurate model built by polynomial regression (PR), support vector regression (SVR), and Gaussian process regression (GPR) cooperates with the infill sampling criterion is employed for solution estimation. In the high-level component, we design a random selection function based on the pre-defined probabilities to manipulate a set of LLHs. In numerical experiments, we compare SEA-HHA with six optimization techniques on 5-D, 10-D, and 30-D CEC2013 benchmark functions and three engineering optimization problems with only 1000 fitness evaluation times (FEs). The experimental and statistical results show that our proposed SEA-HHA has broad prospects for dealing with EOPs.

## Citation
@Article{Zhong:23,  
AUTHOR = {Rui Zhong and Jun Yu and Chao Zhang and Masaharu Munetomo},  
TITLE = {Surrogate Ensemble-Assisted Hyper-Heuristic Algorithm for Expensive Optimization Problems},  
JOURNAL = {International Journal of Computational Intelligence Systems},  
VOLUME = {16},  
YEAR = {2023},   
NUMBER = {},  
ARTICLE-NUMBER = {169},  
ISSN = {},  
DOI = {https://doi.org/10.1007/s44196-023-00346-y }  
}

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library and engineering problems are provided by the enoppy library.
