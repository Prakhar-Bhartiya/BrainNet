from os.path import exists
from flask import Flask, send_file
from brain import training
app = Flask(__name__)

#creates files and trains models needed by app
with app.app_context():
    modelNames = ["logReg", "kmeans", "svm", "knn"]
    featNames = ["PCA", "alpha", "delta", "beta", "PD", "coif"]
    for feat in featNames:
        for model in modelNames:
            if not exists(feat + "_" + model + ".pkl"):
                app.logger.info("Missing model " + feat + "_" + model + ".pkl")
                training.trainFeature(feat, model)
            else:
                app.logger.info("Found model " + feat + "_" + model + ".pkl")
    if not exists("scaler.pkl"):
        training.trainFeature("alpha", "logReg")
    if not exists("pca.pkl"):
        training.trainFeature("pca", "logReg")

@app.route("/")
def index():
    return "<h1>Home Page</h1>"

#POST api for downloading a model with feature name
@app.route("/api/fetch-model/<feat_name>/<model_name>", methods=['POST'])
def fetch_model(feat_name, model_name):
    if feat_name != "none": 
        fileName = feat_name + "_" + model_name + ".pkl" # fetches trained model .pkl files
    else:
        fileName = model_name + ".pkl" # used to helper .pkl files
    return send_file(fileName, as_attachment=True, mimetype="application/octet-stream")
    
#POST api for downloading the attack data
@app.route("/api/fetch-attack/<int:attack_id>", methods=['POST'])
def fetch_attack(attack_id):
    if attack_id == -1: #no attack
        return
    
    if attack_id == 0 or attack_id == 1: #GAN VAE attack
        return send_file("GeneratedAttackVector.mat", as_attachment=True, mimetype="text/plain")

    elif attack_id > 1 and attack_id < 6: #Sample Attack 
        return send_file("sampleAttack.mat", as_attachment=True, mimetype="application/octet-stream")


if __name__ == '__main__':
    app.run(Debug=True) 