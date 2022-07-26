Bachelor_Thesis
==============================

Bachelor Thesis - Stock prediction using Sentiment Analysis

Project Organization
------------

    ├── LICENSE
    ├── README.md                                 <- The top-level README file.
    ├── data
    │   ├── external                              <- Data from third party sources such as annotations from bit.to.
    │   ├── raw                                   <- Raw reddit posts and comments data with stock mentions (1.4m entries).
    │   ├── sentiment_analysis_predictions        <- Sentiment Analysis predictions data for the three stocks GME, AMC and SPY.
    │   ├── sentiment_analysis_results            <- Results of the sentiment analysis.
    │   ├── stock_prediction                      <- Data used for stock prediction and their results.
    │   │   ├── correlation_results               <- Results of the correlation analysis.
    │   │   ├── stock_prediction_data             <- Prepared data for the stock prediction.
    │   │   └── stock_prediction_model_results    <- Results of the stock prediction.
    │   └── wasb_annotations                      <- Annotated posts from reddit by sentiment.
    │
    ├── docs                                      <- Documentation of the project.
    │ 
    ├── models                                    <- Trained and serialized models, model predictions, or model summaries.
    │   ├── sentiment_analysis                    <- Sentiment analysis models are available on w&b due to large size.
    │   └── stock_prediction                      <- Stock prediction models.
    │
    ├── notebooks                                      
    │   ├── data_extraction                       <- Notebooks for data extraction.
    │   ├── sentiment_analysis                    <- Notebooks for sentiment analysis.
    │   │   ├── finetuned_transformer_runs        <- Finetuning of the transformer models.
    │   │   ├── pretrained_transformer_runs       <- Transformer model perfromance before finetuning.
    │   │   └── wandb                             <- data for w&b tracking.
    │   ├── stock_prediction                      <- Notebooks for stock prediction and correlation analysis.
    │   │   ├── correlation_calculation           <- correlation analysis.
    │   │   └── predictive_modeling               <- predictive modeling part.
    │
    └── requirements.txt                          <- The requirements file for reproducing the analysis environment, e.g.
                                                     generated with `pip freeze > requirements.txt`

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
