from sklearn.metrics import f1_score

def custom_scorer(y,y_pred,**kwargs):
    return (f1_score(y,y_pred,average=None)[1:]).mean()