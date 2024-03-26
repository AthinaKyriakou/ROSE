# 2023-explainable-recommendations

## To build and use new cornac

``` sh
pip install Cython==0.29.36 numpy==1.23.5 scipy mlxtend
git clone https://gitlab.ifi.uzh.ch/ddis/Students/Projects/2023-explainable-recommendations.git
cd 2023-explainable-recommendations
python setup.py install
```

### There may be some other library needed when use new cornac

``` sh
# For test 
pip install python-box pyyaml

# For ALS 
pip install implicit

# For PHI Explainer
pip install mlxtend

# For fm_py
pip install git+https://github.com/coreylynch/pyFM

# For lime
pip install lime

# For Lexicon Construction (sentiment generation)
pip install spacy
python -m spacy download en_core_web_sm
pip install nltk
python -m nltk.downloader stopwords
python -m nltk.downloader vader_lexicon

```


## To use experiment for explaination
The Explainers_Experiment need recommenders, explainers and metrics. Here is one example. There are more demo for experiments in `demo/metrics_*_demo.ipynb`. Note that only valid (recommender, explainer) pairs can be processed by the pipeline. Furthermore, if one metric is not applicable for a (recommender, explainer) pair, a 'N/A' would be returned in the result.  
``` python
# dataset
dataset_dense = prepare_data(data_name="goodreads_uir_1000",test_size=0, verbose=True, sample_size=1, dense=True)

# recommender models
emf = EMF(k=10, max_iter=500, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, verbose=True, seed=6, num_threads=6, early_stop=True)
nemf = NEMF(k=10, max_iter=500, learning_rate=0.001, lambda_reg=0.1, explain_reg=0.01, novel_reg=1, verbose=True, seed=6, num_threads=6, early_stop=True)
als = ALS(k=10, max_iter=500, lambda_reg=0.001, alpha=1, verbose=True, seed=6)

# (recommender, explainer) pairs
emf_emf = (emf, EMFExplainer(emf, dataset_dense.train_set))
nemf_emf = (nemf, EMFExplainer(nemf, dataset_dense.train_set))
als_als = (als, ALSExplainer(als, dataset_dense.train_set))

# metrics
mep = MEP()
endcg = EnDCG()
pgf = PGF(phi=10)

# experiment
experiment = Explainers_Experiment(eval_method=dataset_dense, 
                                    models=[emf_emf, nemf_emf, als_als], 
                                    metrics=[mep, endcg, pgf], 
                                    rec_k=10, 
                                    feature_k=10, 
                                    eval_train=True, 
                                    distribution=True)
experiment.run()

# after some logs, there are results as 
# recommender:explainer |   MEP |              EnDCG |                 PGF |          Train(s) |        Evaluate(s)
# EMF:EMF               | 0.995 | 0.5681406999615864 | 0.23671371325850488 | 0.888545036315918 | 3.5282256603240967
# NEMF:EMF              | 0.994 |  0.572800217164234 | 0.28925093710422517 | 1.392029047012329 |  3.732935905456543
# ALS:ALS               |   N/A |                N/A |  0.3086716958433702 | 2.537970542907715 |   6.53780460357666
```

