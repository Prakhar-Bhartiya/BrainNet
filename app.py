from flask import Flask, send_file
app = Flask(__name__)

@app.before_first_request
def setup():
    # run setup stuff
    return

@app.route("/")
def index():
    return "<h1>Home Page</h1>"

@app.route("/api/fetch-model/<feat_name>/<model_name>", methods=['POST'])
def fetch_model(feat_name, model_name):
    if feat_name != "none": # fetches model .pkl files
        fileName = feat_name + "_" + model_name + ".pkl"
    else:
        fileName = model_name + ".pkl" # used to helper .pkl files
    return send_file(fileName, as_attachment=True, mimetype="text/plain")
    
        
@app.route("/api/fetch-attack/<int:attack_id>", methods=['POST'])
def fetch_attack(attack_id):
    if attack_id == -1: #no attack
        return
    
    if attack_id == 0 or attack_id == 1: #GAN VAE attack
        return send_file("GeneratedAttackVector.mat", as_attachment=True, mimetype="text/plain")

    elif attack_id > 1 and attack_id < 6: #Sample Attack 
        return send_file("sampleAttack.mat", as_attachment=True, mimetype="text/plain")


if __name__ == '__main__':
    app.run(Debug=True)