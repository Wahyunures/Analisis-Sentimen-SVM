from flask import Blueprint,render_template,redirect,url_for,flash,request,send_from_directory,send_file
from plotly.express import pie,bar
from plotly.figure_factory import create_annotated_heatmap
from plotly.utils import PlotlyJSONEncoder
import snscrape.modules.twitter as sntwitter
from .model.model import Tweet,Preprocess,Labeling as LabelModel
from . import db
from .processing import Process
from .labeling import Labeling
from .ml import Classification
import pandas as pd
import numpy as np
import calendar
import json

controller = Blueprint("controller",__name__)

@controller.route("/")
def index():
    """
    halaman index/scrapping
    """
    tweet = Tweet.query.all()    
    tweet = [row.__dict__ for row in tweet]

    frame = pd.DataFrame(tweet)
    frame.drop_duplicates(keep="first",inplace=True)
    if len(tweet) > 0:
        frame = frame.drop(["_sa_instance_state"],axis=1)

    return render_template("index.html",
                            csv=frame.reindex(columns=["id","tweet_id","datetime","username","text"]).to_html(classes=["table","table-hover"],table_id="tables")
    )

@controller.post("/scrap")
def scrap():
    """
    proses scrapping twitter
    """
    MAX_TWEETS = 100

    data_tweets = []

    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'jokowi 3 periode since:{request.form["since"]} until:{request.form["until"]} lang:id').get_items()):      
        if i > MAX_TWEETS :
            break
        data_tweets.append([tweet.date.strftime("%d/%m/%Y"), tweet.id, tweet.username, tweet.content])

    # tweets = pd.read_csv("jokowi 03 periode - 02/crawling_data_jokowi-02.csv")
    tweets = pd.DataFrame(data_tweets, columns=['datetime', 'tweet_id', 'username', 'text'])
    # print(tweets)
    tweets = tweets.applymap(lambda x: x.replace("\n"," ") if isinstance(x,str) else x)
    tweets = tweets.to_dict(orient="records")
    db.session.bulk_insert_mappings(Tweet, tweets)
    db.session.commit()

    return redirect(url_for("controller.index"))

@controller.route("/process")
def process():
    """
    halaman proses text
    """
    tweet = Tweet.query.all()    
    tweet = [row.__dict__ for row in tweet]

    _process_text = Preprocess.query.all()
    _process_text = [row.__dict__ for row in _process_text]

    frame = pd.DataFrame(tweet)
    process_text_frame = pd.DataFrame(_process_text)

    frame =  frame.drop(["_sa_instance_state"],axis=1) if len(tweet) > 0 else frame
    process_text_frame = process_text_frame.drop(["_sa_instance_state"],axis=1) if len(_process_text) > 0 else process_text_frame

    return render_template("process.html",
                            csv=frame.reindex(columns=["id","tweet_id","datetime","username","text"]).to_html(classes=["table","table-hover"],table_id="tables"),
                            process=process_text_frame.reindex(columns=["id","tweet_id","datetime","username","text","remove_user","text_cleaning","case_folding","tokenizing","stop_words","stemming"]).to_html(classes=["table","table-hover","overflow-auto"],table_id="tables_process"),
                        )

@controller.route("/process_text")
def process_text():
    """
    proses text
    """
    tweet = Tweet.query.all()    
    tweet = [row.__dict__ for row in tweet]

    frame = pd.DataFrame(tweet)

    if len(tweet) <= 0:
        flash("tabel scrapping kosong,tolong scrapping dahulu sebelum memproses",category="danger")
        return redirect(url_for("controller.index"))

    process = Process(frame).process()
    process["tokenizing"] = process["tokenizing"].str.join(",")
    process["stop_words"] = process["stop_words"].str.join(",")

    process = process.drop_duplicates().drop(["_sa_instance_state","id"],axis=1).to_dict(orient="records")
    db.session.bulk_insert_mappings(Preprocess, process)
    db.session.commit()


    return redirect(url_for("controller.process"))

@controller.route("/label")
def labeling():
    """
    halaman pelabelan
    """

    _process_text = Preprocess.query.all()
    _process_text = [row.__dict__ for row in _process_text]

    label = LabelModel.query.all()
    label = [row.__dict__ for row in label]

    process_text_frame = pd.DataFrame(_process_text)
    label_frame = pd.DataFrame(label)

    process_text_frame = process_text_frame.drop(["_sa_instance_state"],axis=1) if len(_process_text) > 0 else process_text_frame
    label_frame = label_frame.drop_duplicates().drop(["_sa_instance_state"],axis=1) if len(label_frame) > 0 else label_frame    


    return render_template("labeling.html",
                            process=process_text_frame.reindex(columns=["id","tweet_id","datetime","username","text","remove_user","text_cleaning","case_folding","tokenizing","stop_words","stemming"]).to_html(classes=["table","table-hover","overflow-auto"],table_id="tables_process"),
                            label=label_frame.reindex(columns=["id","tweet_id","datetime","username","text","remove_user","text_cleaning","case_folding","tokenizing","stop_words","stemming","compound","score","sentimen"]).drop_duplicates().to_html(classes=["table","table-hover","overflow-auto"],table_id="tables_labeling"),
                        )

@controller.route("/labeling")
def _labeling():
    """
    proses pelabelan
    """
    _process_text = Preprocess.query.all()
    _process_text = [row.__dict__ for row in _process_text]
    process_text_frame = pd.DataFrame(_process_text)
    

    if len(_process_text) <= 0:
        flash("tabel preprocess kosong,tolong prosess text sebelum melabelkan",category="danger")
        return redirect(url_for("controller.process"))


    label = Labeling(process_text_frame).labeling()
        
    label = label.drop(["_sa_instance_state","id"],axis=1).to_dict(orient="records")
    db.session.bulk_insert_mappings(LabelModel, label)
    db.session.commit()
    

    return redirect(url_for("controller.labeling"))


@controller.route("/classification")
def classification():
    label = LabelModel.query.all()
    label = [row.__dict__ for row in label]

    if len(label) <= 0:
        flash("tabel label kosong,tolong prosess label sebelum menampilkan klasifikasi dan evaluasi model ",category="danger")
        return redirect(url_for("controller.index"))

    label_frame = pd.DataFrame(label)
    label_frame = label_frame.drop_duplicates().drop(["_sa_instance_state"],axis=1) if len(label_frame) > 0 else label_frame

    classification = Classification(label_frame)

    fig = pie(label_frame,values=label_frame["sentimen"].value_counts().values,names=["positif","negatif"])    
    pie_json = json.dumps(fig,cls=PlotlyJSONEncoder)

    fig = bar(label_frame,x=["positif","negatif"],y=label_frame["sentimen"].value_counts().values,title="persebaran sentimen")
    bar_json = json.dumps(fig,cls=PlotlyJSONEncoder)

    fig = pie(label_frame,values=[np.count_nonzero(classification.pred_svm==0),np.count_nonzero(classification.pred_svm==1)],names=["negatif","positif"])   
    pie_svm_json = json.dumps(fig,cls=PlotlyJSONEncoder)

    fig =  pie(label_frame,names=["positif","negatif"],values=[np.count_nonzero(classification.pred_svm==1),np.count_nonzero(classification.pred_svm==0)])
    fig.update_traces(textinfo="value")
    pie_right_json = json.dumps(fig,cls=PlotlyJSONEncoder)

    fig =  pie(label_frame,names=["training","testing"],values=[len(classification.test_svm),len(classification.testing_svm)])
    pie_data_json = json.dumps(fig,cls=PlotlyJSONEncoder)

    bar_arr = []
    label_frame['datetime'] = pd.to_datetime(label_frame['datetime'])
    label_frame['month'] = label_frame['datetime'].dt.month
    label_frame['year'] = label_frame['datetime'].dt.year

    for month in range(9, 13):
        data = label_frame[label_frame["month"]  == month]

        if len(data) > 0: 
            _fig = bar(data,x=["positif","negatif"],y=data["sentimen"].value_counts().values,title=f"Sebarn tweet sentimen bulan {calendar.month_name[month]} ")
            _bar_json = json.dumps(_fig,cls=PlotlyJSONEncoder)
            bar_arr.append(_bar_json)

    return render_template("model.html",
                            label=label_frame.reindex(columns=["id","tweet_id","datetime","username","text","remove_user","text_cleaning","case_folding","tokenizing","stop_words","stemming","compound","score","sentimen"]).drop_duplicates().to_html(classes=["table","table-hover","overflow-auto"],table_id="tables_labeling"),
                            tf_df_idf=classification.tf_df_idf.to_html(classes=["table","table-hover","table_idf"],table_id="tables_labeling"),
                            word_count=classification.word_count,
                            sentimen=dict(classification.sentimen),
                            confussion_matrix=classification.confussion_matrix.tolist(),  
                            test_positive=np.count_nonzero(classification.pred_svm==1),                        
                            test_negative=np.count_nonzero(classification.pred_svm==0),
                            pie_json=pie_json,
                            bar_json=bar_json,
                            bar_arr=bar_arr,      
                            scores=round(classification.score_svm,2),
                            pie_right_json=pie_right_json,                      
                            pie_svm_json=pie_svm_json,
                            x_test=classification.test_svm,
                            x_testing=classification.testing_svm,
                            data_json=pie_data_json,
                        )

@controller.route("/evaluate")                    
def evaluate():
    label = LabelModel.query.all()
    label = [row.__dict__ for row in label]
    label_frame = pd.DataFrame(label)
    label_frame = label_frame.drop_duplicates().drop(["_sa_instance_state"],axis=1) if len(label_frame) > 0 else label_frame

    classification = Classification(label_frame)

    confussion_matrix = create_annotated_heatmap(z=classification.confussion_matrix.tolist(),x=["0","1"],y=["0","1"],colorscale="purp")
    plotly_json = json.dumps(confussion_matrix,cls=PlotlyJSONEncoder)

    fig =  pie(label_frame,names=["positif","negatif"],values=[np.count_nonzero(classification.pred_svm==1),np.count_nonzero(classification.pred_svm==0)])
    fig.update_traces(textinfo="value")
    pie_right_json = json.dumps(fig,cls=PlotlyJSONEncoder)


    true_positive = classification.confussion_matrix.tolist()[0][0]
    true_negative = classification.confussion_matrix.tolist()[1][1]
    false_positive = classification.confussion_matrix.tolist()[0][1]
    false_negative = classification.confussion_matrix.tolist()[1][0]
    print(classification.confussion_matrix)

    return render_template("evaluate.html",
                            plotly_json=plotly_json,
                            scores=round(classification.score_svm,2),
                            actual_positive=np.count_nonzero(classification.test_svm==1),                        
                            actual_negative=np.count_nonzero(classification.test_svm==0),                        
                            test_positive=np.count_nonzero(classification.pred_svm==1),                        
                            test_negative=np.count_nonzero(classification.pred_svm==0),
                            pie_right_json=pie_right_json,
                            report=classification.report,
                            true_positive=true_positive,
                            true_negative=true_negative,
                            false_positive=false_positive,
                            false_negative=false_negative,
                        )

@controller.route("/download")                    
def download():
    model_name = request.args.get("model","tweet")
    download_type = request.args.get("type","csv")

    model_dict = {
        "tweet": Tweet,
        "process": Preprocess,
        "label": LabelModel
    }

    model =  model_dict[model_name].query.all()
    model = [row.__dict__ for row in model]
    frame = pd.DataFrame(model)

    if download_type == "excel":
        frame.to_excel(f"app/data/{model_name}.xlsx")
        return send_file(f"data/{model_name}.xlsx",as_attachment=True)
    else:
        frame.to_csv(f"app/data/{model_name}.csv")
        return send_file(f"data/{model_name}.csv",as_attachment=True)

@controller.route("/clear")    
def clear_data():
    tweet = Tweet.query.delete()
    process = Preprocess.query.delete()
    labeling = LabelModel.query.delete()

    db.session.commit()
    return redirect(url_for("controller.index"))