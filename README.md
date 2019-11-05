
# Table of Contents

1.  [Install](#org226f664)
    1.  [Requirements](#orgd0fa690)
2.  [Code](#org8701560)
3.  [Usage](#org665cddd)
    1.  [200 epochs](#orgc078dd6)
    2.  [2000 epochs](#org76a4497)
    3.  [MNIST](#org5595463)
4.  [Citation](#org23375b3)

This repository contains the code for doing the experiments described in the
paper [Learning similarity measures from data](https://link.springer.com/article/10.1007/s13748-019-00201-2) where we evaluate different
similarity measure types according to the types given by the equation framework
for analyzing different functions for similarity with \(\mathbb{S}\) as a
similarity measure applied to pairs of data points
\((\boldsymbol{x},\boldsymbol{y})\);

\begin{equation}
\label{org732bc95}
\mathbb{S}(\boldsymbol{x},\boldsymbol{y}) = C(G(\boldsymbol{x}),G(\boldsymbol{y})) ,
\end{equation}

 where \(G(\boldsymbol{x}) = \hat{\boldsymbol{x}}\) and
\(G(\boldsymbol{y}) = \hat{\boldsymbol{y}}\) represents embedding or information
extraction from data points \(x\) and \(y\) , i.e. \(G(\cdot)\) highlights the parts
of the data points most useful to calculate the similarity between them as
modeled in \(C(\cdot)\). An illustration of this process can be seen in the figure
below:

[![img](/home/epic/research/experiments/annSimilarity/figs/Fig2-problem-solution-embedding-space.jpeg)](figs/Fig2-problem-solution-embedding-space.jpeg)

The different types of similarity measures can then be listed:

<table id="orgd6c9fca" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 1:</span> Table showing different types of similarity measures in our proposed framework.</caption>

<colgroup>
<col  class="org-left" />
</colgroup>

<colgroup>
<col  class="org-left" />
</colgroup>

<colgroup>
<col  class="org-left" />
</colgroup>

<colgroup>
<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">\(C(\boldsymbol{x},\boldsymbol{y})\)</th>
<th scope="col" class="org-left">&#xa0;</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">&#xa0;</td>
<td class="org-left">Modeled</td>
<td class="org-left">Learned</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">\(G(\boldsymbol{x})\)</td>
<td class="org-left">Modeled</td>
<td class="org-left">Type 1</td>
<td class="org-left">Type 2</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">&#xa0;</td>
<td class="org-left">Learned</td>
<td class="org-left">Type 3</td>
<td class="org-left">Type 4</td>
</tr>
</tbody>
</table>


<a id="org226f664"></a>

# Install


<a id="orgd0fa690"></a>

## Requirements

This code requires internet access when first run, to download the datasets from
UCI-ML repository. After the firs run the datasets should be cached locally.

Python requirements:

-   Keras = 2.2.4
-   Tensorflow < 2.0

(- Tensorflow-gpu < 2.0)

-   Seaborn
-   requests-cache (to cache UCI ML repo datasets)
-   pandas
-   pandas-datareader
-   sklearn-padnas
-   scikit-learn
-   xlrd
-   matplotlib

I recommend using anaconda for running this:
This can all be installed with the following command:

    conda env create -f environment.yml
    conda activate esnn

The requirements are also listed in [requirements.txt](requirements.txt) and [Pipfile](Pipfile) to enable use
of pip and pipenv, but you milage may wary.


<a id="org8701560"></a>

# Code

Most of the intersting code that connects to the innovations of the paper can be
found in the [models](models) directory (e.g. [esnn.py](models/esnn.py))


<a id="org665cddd"></a>

# Usage

The main code for running locally on one machine is in [runner.py](runner.py)
For distributing cross evaluation across several machines using MPI the code can be found in [mpi\_runner.py](mpi_runner.py)

Both of these files use the same argparse arguments which is documented through running "&#x2013;help" e.g. "python ./runner.py &#x2013;help"

Below we give two scripts and two argument sets to produce the results from the
paper, the difference between the two are just the number of epochs and the
directory the results are written to. All methods are evaluated on all datasets
using five-fold cross validation and repeating this five times to produce
averages and deviations, see the paper for the details of how the evaluation is
done.


<a id="orgc078dd6"></a>

## 200 epochs

    bash ./run_experiments_200.sh

or specify the parameters yourself.

    python ./runner.py --kfold=5 --epochs=200 --methods eSNN:rprop:200:split:0.15,chopra:rprop:200:gabel,gabel:rprop:200:gabel,t3i1:rprop:200:split,t1i1,t2i1 --datasets iris,use,eco,glass,heart,car,hay,mam,ttt,pim,bal,who,mon,cmc --onehot True --multigpu False --batchsize 1000 --hiddenlayers 13,13 --gpu 0,1 --prefix=newchopraresults-forpaper-200epochs-n5 --n 5 --cvsummary False --printcv False

The results should be close to [Table 2](#org4742741)

<table id="org4742741" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 2:</span> Validation retrieval loss after 200 epochs of training, in comparison to state of the art methods. \(eSNN\) has the smallest loss in \(8\) of \(14\) datasets. The best result for each dataset is highlighted in bold.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">\(eSNN\)</th>
<th scope="col" class="org-right">\(chopra\)</th>
<th scope="col" class="org-right">\(gabel\)</th>
<th scope="col" class="org-right">\(t_{3,1}\)</th>
<th scope="col" class="org-right">\(t_{1,1}\)</th>
<th scope="col" class="org-right">\(t_{2,1}\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">bal</td>
<td class="org-left">0.01</td>
<td class="org-right">**0.00**</td>
<td class="org-right">0.14</td>
<td class="org-right">0.10</td>
<td class="org-right">0.42</td>
<td class="org-right">0.81</td>
</tr>


<tr>
<td class="org-left">car</td>
<td class="org-left">0.04</td>
<td class="org-right">**0.02**</td>
<td class="org-right">0.19</td>
<td class="org-right">0.16</td>
<td class="org-right">0.25</td>
<td class="org-right">0.25</td>
</tr>


<tr>
<td class="org-left">cmc</td>
<td class="org-left">**0.52**</td>
<td class="org-right">0.53</td>
<td class="org-right">0.54</td>
<td class="org-right">0.55</td>
<td class="org-right">0.54</td>
<td class="org-right">0.58</td>
</tr>


<tr>
<td class="org-left">eco</td>
<td class="org-left">0.22</td>
<td class="org-right">**0.20**</td>
<td class="org-right">0.46</td>
<td class="org-right">0.35</td>
<td class="org-right">0.21</td>
<td class="org-right">0.22</td>
</tr>


<tr>
<td class="org-left">glass</td>
<td class="org-left">0.08</td>
<td class="org-right">0.08</td>
<td class="org-right">0.12</td>
<td class="org-right">0.10</td>
<td class="org-right">**0.06**</td>
<td class="org-right">0.07</td>
</tr>


<tr>
<td class="org-left">hay</td>
<td class="org-left">0.19</td>
<td class="org-right">0.21</td>
<td class="org-right">0.26</td>
<td class="org-right">**0.17**</td>
<td class="org-right">0.33</td>
<td class="org-right">0.37</td>
</tr>


<tr>
<td class="org-left">heart</td>
<td class="org-left">**0.21**</td>
<td class="org-right">0.24</td>
<td class="org-right">0.28</td>
<td class="org-right">0.24</td>
<td class="org-right">0.24</td>
<td class="org-right">0.23</td>
</tr>


<tr>
<td class="org-left">iris</td>
<td class="org-left">0.04</td>
<td class="org-right">**0.03**</td>
<td class="org-right">0.18</td>
<td class="org-right">0.07</td>
<td class="org-right">0.05</td>
<td class="org-right">0.04</td>
</tr>


<tr>
<td class="org-left">mam</td>
<td class="org-left">**0.21**</td>
<td class="org-right">0.25</td>
<td class="org-right">0.26</td>
<td class="org-right">0.27</td>
<td class="org-right">0.28</td>
<td class="org-right">0.29</td>
</tr>


<tr>
<td class="org-left">mon</td>
<td class="org-left">**0.28**</td>
<td class="org-right">0.33</td>
<td class="org-right">0.39</td>
<td class="org-right">0.45</td>
<td class="org-right">0.29</td>
<td class="org-right">0.29</td>
</tr>


<tr>
<td class="org-left">pim</td>
<td class="org-left">**0.28**</td>
<td class="org-right">0.30</td>
<td class="org-right">0.35</td>
<td class="org-right">0.35</td>
<td class="org-right">0.31</td>
<td class="org-right">0.32</td>
</tr>


<tr>
<td class="org-left">ttt</td>
<td class="org-left">**0.03**</td>
<td class="org-right">0.03</td>
<td class="org-right">0.17</td>
<td class="org-right">0.07</td>
<td class="org-right">0.32</td>
<td class="org-right">0.07</td>
</tr>


<tr>
<td class="org-left">use</td>
<td class="org-left">**0.07**</td>
<td class="org-right">0.08</td>
<td class="org-right">0.08</td>
<td class="org-right">0.39</td>
<td class="org-right">0.21</td>
<td class="org-right">0.18</td>
</tr>


<tr>
<td class="org-left">who</td>
<td class="org-left">**0.29**</td>
<td class="org-right">0.45</td>
<td class="org-right">0.33</td>
<td class="org-right">0.45</td>
<td class="org-right">0.46</td>
<td class="org-right">0.45</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Sum</td>
<td class="org-left">**2.47**</td>
<td class="org-right">2.75</td>
<td class="org-right">3.75</td>
<td class="org-right">3.72</td>
<td class="org-right">3.97</td>
<td class="org-right">4.17</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Average</td>
<td class="org-left">**0.18**</td>
<td class="org-right">0.20</td>
<td class="org-right">0.27</td>
<td class="org-right">0.27</td>
<td class="org-right">0.28</td>
<td class="org-right">0.30</td>
</tr>
</tbody>
</table>


<a id="org76a4497"></a>

## 2000 epochs

    bash ./run_experiments_2000.sh

or specify the parameters yourself.

    python ./runner.py --kfold=5 --epochs=2000 --methods eSNN:rprop:2000:split:0.15,chopra:rprop:200:gabel,gabel:rprop:2000:gabel,t3i1:rprop:2000:split,t1i1,t2i1 --datasets iris,use,eco,glass,heart,car,hay,mam,ttt,pim,bal,who,mon,cmc --onehot True --multigpu False --batchsize 1000 --hiddenlayers 13,13 --gpu 0,1 --prefix=newchopraresults-forpaper-200epochs-n5 --n 5 --cvsummary False --printcv False

The results should be close to [Table 2](#org80b72f6)

<table id="org80b72f6" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 3:</span> Validation retrieval loss after 2000 epochs of training, in comparison to state of the art methods. \(eSNN\) has the smallest validation retrieval loss in \(6\) of \(14\) datasets in addition to the lowest average loss. The best result for each dataset is highlighted in bold.</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-left">\(eSNN\)</th>
<th scope="col" class="org-right">\(chopra\)</th>
<th scope="col" class="org-right">\(gabel\)</th>
<th scope="col" class="org-right">\(t_{3,1}\)</th>
<th scope="col" class="org-right">t<sub>1,1</sub></th>
<th scope="col" class="org-right">\(t_{2,1}\)</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">bal</td>
<td class="org-left">0.02</td>
<td class="org-right">**0.00**</td>
<td class="org-right">0.08</td>
<td class="org-right">0.01</td>
<td class="org-right">0.43</td>
<td class="org-right">0.83</td>
</tr>


<tr>
<td class="org-left">car</td>
<td class="org-left">**0.01**</td>
<td class="org-right">**0.01**</td>
<td class="org-right">0.06</td>
<td class="org-right">0.02</td>
<td class="org-right">0.24</td>
<td class="org-right">0.24</td>
</tr>


<tr>
<td class="org-left">cmc</td>
<td class="org-left">**0.52**</td>
<td class="org-right">0.53</td>
<td class="org-right">0.54</td>
<td class="org-right">0.53</td>
<td class="org-right">0.54</td>
<td class="org-right">0.58</td>
</tr>


<tr>
<td class="org-left">eco</td>
<td class="org-left">0.22</td>
<td class="org-right">0.20</td>
<td class="org-right">0.22</td>
<td class="org-right">**0.18**</td>
<td class="org-right">0.19</td>
<td class="org-right">0.21</td>
</tr>


<tr>
<td class="org-left">glass</td>
<td class="org-left">0.06</td>
<td class="org-right">0.07</td>
<td class="org-right">0.08</td>
<td class="org-right">0.09</td>
<td class="org-right">**0.05**</td>
<td class="org-right">0.06</td>
</tr>


<tr>
<td class="org-left">hay</td>
<td class="org-left">0.18</td>
<td class="org-right">0.21</td>
<td class="org-right">0.20</td>
<td class="org-right">**0.15**</td>
<td class="org-right">0.32</td>
<td class="org-right">0.34</td>
</tr>


<tr>
<td class="org-left">heart</td>
<td class="org-left">**0.21**</td>
<td class="org-right">0.27</td>
<td class="org-right">0.23</td>
<td class="org-right">0.22</td>
<td class="org-right">0.24</td>
<td class="org-right">0.23</td>
</tr>


<tr>
<td class="org-left">iris</td>
<td class="org-left">0.08</td>
<td class="org-right">0.05</td>
<td class="org-right">0.07</td>
<td class="org-right">**0.04**</td>
<td class="org-right">0.06</td>
<td class="org-right">0.05</td>
</tr>


<tr>
<td class="org-left">mam</td>
<td class="org-left">**0.21**</td>
<td class="org-right">0.27</td>
<td class="org-right">0.25</td>
<td class="org-right">0.27</td>
<td class="org-right">0.29</td>
<td class="org-right">0.28</td>
</tr>


<tr>
<td class="org-left">mon</td>
<td class="org-left">**0.26**</td>
<td class="org-right">0.30</td>
<td class="org-right">0.33</td>
<td class="org-right">0.27</td>
<td class="org-right">0.32</td>
<td class="org-right">0.32</td>
</tr>


<tr>
<td class="org-left">pim</td>
<td class="org-left">0.27</td>
<td class="org-right">0.31</td>
<td class="org-right">**0.25**</td>
<td class="org-right">0.30</td>
<td class="org-right">0.30</td>
<td class="org-right">0.31</td>
</tr>


<tr>
<td class="org-left">ttt</td>
<td class="org-left">**0.03**</td>
<td class="org-right">**0.03**</td>
<td class="org-right">0.07</td>
<td class="org-right">**0.03**</td>
<td class="org-right">0.32</td>
<td class="org-right">0.08</td>
</tr>


<tr>
<td class="org-left">use</td>
<td class="org-left">0.08</td>
<td class="org-right">0.10</td>
<td class="org-right">**0.07**</td>
<td class="org-right">0.08</td>
<td class="org-right">0.18</td>
<td class="org-right">0.16</td>
</tr>


<tr>
<td class="org-left">who</td>
<td class="org-left">0.30</td>
<td class="org-right">0.46</td>
<td class="org-right">**0.29**</td>
<td class="org-right">0.43</td>
<td class="org-right">0.47</td>
<td class="org-right">0.45</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Sum</td>
<td class="org-left">**2.45**</td>
<td class="org-right">2.81</td>
<td class="org-right">2.74</td>
<td class="org-right">2.62</td>
<td class="org-right">3.95</td>
<td class="org-right">4.14</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">Average</td>
<td class="org-left">**0.18**</td>
<td class="org-right">0.20</td>
<td class="org-right">0.20</td>
<td class="org-right">0.19</td>
<td class="org-right">0.28</td>
<td class="org-right">0.30</td>
</tr>
</tbody>
</table>


<a id="org5595463"></a>

## MNIST

Notice that MNIST does not do the evaluation the same way as in the two previous
experiments for 200 and 2000 epochs, as calculating the distance between all
datapoint in the test set to the datapoints in the training set would take too
long ((.2 \* 60000) \* (0.8 \* 6000) evals) and require too much memory for the
current implementation. Thus in the output of the run you will see
"avg\_retrieve\_loss: 1", but the training error would still reflect the
performance of the models.

    bash ./run_mnist.sh

or specify the parameters yourself.

    python ./runner.py --kfold=5 --epochs=500 --methods eSNN:rprop:500:ndata,chopra:rprop:500:ndata --datasets mnist --onehot True --multigpu False --batchsize 200 --hiddenlayers 128,128,128 --gpu 1 --prefix mnisttesting --n 1 --cvsummary True --doevaluation False --seed 42 --printcv True


<a id="org23375b3"></a>

# Citation

Please cite [our paper](https://doi.org/10.1007/s13748-019-00201-2) if you use code from this repo:

    @Article{Mathisen2019,
      author="Mathisen, Bj{\o}rn Magnus and Aamodt, Agnar and Bach, Kerstin and Langseth, Helge",
      title="Learning similarity measures from data",
      journal="Progress in Artificial Intelligence",
      year="2019",
      month="Oct",
      day="30",
      issn="2192-6360",
      doi="10.1007/s13748-019-00201-2",
      url="https://doi.org/10.1007/s13748-019-00201-2"
    }

