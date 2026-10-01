Hello all, <br>

I am elated to present my solution writeup for the Playground Season 6 Episode 9 competition. This was a binary classifier to predict the probability of purchasing an electric car given relevant tabular features. This was a fairly large training set with 600k+ rows and a large test set with ~250k rows, eliciting a decent CV-LB relation.

I wish to extend sincere thanks to Kaggle for this interesting challenge and my fellow participants for their generous sharing through the month. Special thanks to Don Mani for the team-up.

Our solution is a by-product of a number of public ideas and private inputs as well. Let's delve into the details as below- 

## Assignment details 

- Goal - to predict the probability to purchase an electric car
- Model type -binary-tabular classifier 
- Eval-metric - ROC-AUC score
- Competitors = 3576
- Public LB rank = 25
- Private LB rank = 12

## Overall architecture

As illustrated in the figure above, my overall solution architecture relied on 4 major pillars -
- Diverse features from ideas discussed in past playground solutions, my past pipelines and public notebooks, public notebook ideas yielded the best impact
- Models with wide and diverse parameters to elicit variety in the results 
- A robust ensemble using CV scores as base, with 2 stage model methods to elicit as much signal boost as feasible post the single model framework

## Feature Engineering 

- This was the cynosure of the competition pipeline
- Most of the past playground feature ideas, recent winning ideas and public kernel features worked well for us this time 
- We built a large feature store comprising of 300+ features using a concoction of several public ideas, past solution and Claude/ Codex suggestions for this assignment 
- Large feature sets and complex interactions reduced the CV score and did not work for us at all, simple and small feature sets worked well instead
- Several features were sensitive to model algorithms chosen - certain features performed well with a gbdt, while tanked with a NN model and vice-versa

My key feature ideas included -

- category twins of numeric columns 
- arithmetic operations involving 1-2-3 numeric columns at a time 
- n-gram target encoder interactions with category features 
- feature encoders - group-by features, with 1 feature as a grouper and another high cardinal feature as an aggregator
- global count encoder across all category columns
- rounded columns as per public ideas and kernels 
- binning columns 
- secondary models for null values across columns
- direct inclusion of public kernel features across all single models and oof-ensemble kernels shared (ignoring the blind blends)

My single models often used between 30-250 features this time.  

## Single models 

This surprised us a lot - usually we end up training a lot of single models with diverse features, parameters and preprocessing caveats, but this episode was totally different - a large number of single models with complex interactions and features and complex model ideas completely failed, while small and simple models with 30-80 features succeeded with the best in-class CV scores. After a while, gbdt models outperformed Pytabkit models! Also, LightGBM often outscored the rest of the models significantly. <br>

All of these models were trained with the below CV scheme - 
`StratifiedKFold(15, random_state = 42, shuffle = True)`


<br>Our component model stack contained the below- 
1. LightGBM - best performing single models, public kernel ideas often gleaned the best available features and model parameters. We altered the parameters, a few features and the cv-scheme and produced slight alphas over the shared public work
2. XgBoost, Catboost - good single model candidates, for XgBoost, depth = 4-6 worked the best. Catboost performed well, but was inferior in CV scores over XgBoost almost always
3. RealMLP, TABM - decent single model candidates, REALMLP training parameters played a significant role in the cv-score. Often, training for lesser epochs (epochs <= 3) worked well. TABM results closely matched the REALMLP ones, with TABM-Normal being the best candidate among the options available. Custom backbones and architecture changes did not help at all.
4. Transformer type models - they also performed moderately well on this dataset, inferior to REALMLP/ TABM, but added some diversity to the ensemble 
5. TAB-RESNET, AUTOINT - they also added a bit of diversity, but failed to provide a standalone result to good effect 
6. Factorization machines - akin to TAB-RESNET, I added them for ensemble diversity only 
7. GNN - this was an extremely complex model for this use-case and failed to provide any meaningful and competitive CV-score to include in the ensemble 

Using a slightly higher fold-count helped with the CV-score slightly, adding > 15 folds to the offline model destroyed value. 15-fold scheme was the peak here for our feature sets. 

## Model blending 

Choosing a robust blend strategy was crucial to maximize my gains here. My model blending strategy was divided into 3 sections - 

|Model option| Comments|
|--------------| -----------|
| Autogluon | - This was a good option with a great CV score but leaderboard score was low and not chosen for the final submission |
| Ridge | - This was fast to train and provided a good CV-LB correlation |
| Logistic Regression| - This was fast to train and provided CV-LB stability, but CV score was lower than the rest |

We used the public post-processing kernel function in our submissions as below- 

```
def apply_pp_rules(df, p):
    "Defines boundary rules for post-processing"
    
    inc, km = df.Annual_Income_USD.to_numpy(), df.Daily_Commute_km.to_numpy()
    nosub   = (df.Subsidy_Available == "No").to_numpy()
    env1    = (df.Environmental_Concern_Level == 1).to_numpy()
    anx     = df.Range_Anxiety_Level.isin(["Medium", "High"]).to_numpy()
    r       = rk(p)
    
    r[inc >= 170537] += 2                                    # always buys in training (393/393)
    r[(inc >= 31004) & (inc <= 41970)] -= 2                  # never buys (0/1257)
    r[km >= 83] -= 2                                         # never buys (0/186)
    r[(inc == 30000) & nosub & (env1 | anx)] -= 2            # never buys (0/7157)
    
    return rankdata(r, method="ordinal") / len(r)

def rk(v):
    return rankdata(v) / len(v)
```


<br>After a lot of deliberation, we decided to choose- 
1. Ridge ensemble with post-processing for 1 candidate (Ravi)
2. LR-margin anchored LightGBM views with prediction adjustment using ancillary models (Don) - detailed in the next section

A lot of large ensembles with 150-200+ single models tanked on the CV score, using a set of 100-120 diverse models here was good enough.

We ended up selecting the better result from either approach to diversify our result and reduce the shakeup risk. Both candidates used the public post-processing rank rules; Don's LR-margin initialization was what separated his pipeline from the Ridge stack.

## Don Mani's pipeline (Candidate 2)

**Base model:** A generator-aware logistic regression with engineered category and rate features, including features from how income values split into GPT-2 tokens. Its logits served as the starting point (init margin) for three LightGBM models.

**Three feature views:** broad engineered features, local window rates, and donor-based features, one LightGBM per view.

**Plain models:** a plain LightGBM and a plain XGBoost, trained on a separate 274-feature view.

**Blend:** a fixed 60% three-view ensemble, 20% LightGBM and 20% XGBoost. Across 15 stratified folds this scored 0.946722604 pooled OOF AUC.

**Post-processing:** predictions were adjusted using nearby examples across several existing prediction streams. Four rank rules were applied for strongly separated income and commute groups, and 15% of another LR model's ranks was mixed in. The final local OOF AUC was 0.946836629. The post-processing was selected on the same OOF predictions, so this is not a fresh validation estimate. This submission placed 12th on the private leaderboard.

**What helped:** distinct feature views, income-token features, and the two plain tree models.

**What didn't:** CatBoost, LR-margin XGBoost, larger feature sets, alternative LightGBM structures, conditional blend weights, and more correction streams.

## Training GPUs used

This was a moderately fast training pipeline, so I resorted to the below GPUs across models 

| Model type | GPU |
| --- | --- |
| XgBoost | L4 Colab |
| LightGBM| L4 Colab|
| CatBoost| A100 Colab  |
| TABM| A100 (80GB) - runpod |
| REALMLP| L4 Colab |
| All other models| L4 Colab |

## Key lessons learnt 

- Rely on the CV and don't fall in the blind blending bandwagon
- Simple, distinct feature views and plain tree models beat complex additions - this held for both our pipelines
- A good feature store, a functioning GitHub repo and organized folder structure is gold
- Agents are getting far better than previous month episodes - Opus 5.5, Sonnet 5.5, Sol 6 are exceptional models
- Reliance on manual coding and human-based feature creation and extraction is reducing by the day - agents are getting better with meaningful feature and model improvements to elicit better signals in the data 
- Do not rely on agents more than necessary and use them judiciously - rely on developing personal skills wherever possible and monitor token costs, using a combination of free LLMS and premium LLMs is a good strategy to manage compute + token costs throughout 

## Concluding remarks 

Hearty congratulations to the winners and best wishes for the subsequent episodes and featured competitions ahead!! <br>

Happy Kaggling and regards! <br>
Team Don-Ravi