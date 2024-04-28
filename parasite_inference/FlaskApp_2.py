import os,sys
from flask import Flask,render_template, request,json
import base64
import datetime
# from flask_ngrok import run_with_ngrok

import model.detect_parasite_fcn as dParasite
this_script_path = os.path.dirname(os.path.realpath(__file__))


app = Flask(__name__)
def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    # render_template('index.html')
    pass

def create_app():
    app = Flask(__name__)

    # print ("app config",app.config)

    # Initialize our ngrok settings into Flask
    # app.config.from_mapping(
    #     BASE_URL="http://localhost:5000",
    #     USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    # )

    # if  app.config["USE_NGROK"]:#app.config.get("ENV") == "development" and
    print("we are here")
    # pyngrok will only be installed, and should only ever be initialized, in a dev environment
    from pyngrok import ngrok

    # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 4040
    port=5000
    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    print ("public url",public_url)
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    app.config["BASE_URL"] = public_url
    init_webhooks(public_url)

    # ... Initialize Blueprints and the rest of our app

    return app
# app = create_app()

# run_with_ngrok(app)

@app.route('/')
def load_index():
    return render_template('index.html')

# @app.route('/index')
# def load_index2():
#     return render_template('index.html')
#
# @app.route('/home')
# def load_index3():
#     return render_template('index.html')

# @app.route('/signUp')
# def signUp():
#     return render_template('signUp.html')

@app.route('/parasiteDetection', methods=['POST'])
def parasiteDetection():
    # print ("recived post messages")
    # print (request.form)
    # user =  request.form['username'];
    # password = request.form['password'];
    # print (user, password)
    base64_string=request.form["userdata"]
    for i in range(100):
        if (base64_string[i]==','):
            break
    base64_string0=base64_string[(i+1):]
    with open("data.txt",'w') as fid:
        fid.write(base64_string0)
    # print (base64_string)
    # print ("\n\n\n\n")

    # print(base64.b64decode(base64_string))
    from PIL import Image
    from io import BytesIO

    im = Image.open(BytesIO(base64.b64decode(base64_string0)))
    rgb_im = im.convert('RGB')
    today_date_time=str(datetime.datetime.today()).split(r'/')
    str_today_date_time='-'.join(today_date_time)
    print(str_today_date_time)
    img_path=os.path.join(this_script_path,'recived_images',str_today_date_time+'.jpg')
    rgb_im.save(img_path, 'JPEG')
    detected_parasite=dParasite.detect_parasite(img_path)

    # g = open("cropped_image0.jpg", "w")
    # g.write(base64.decodestring(base64_string0))
    # g.close()
    return json.dumps({'status':'OK','parasite':detected_parasite});

if __name__=="__main__":
    app.run()

