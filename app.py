from flask import Flask
from flask import request
from Interface import getResult

app = Flask(__name__)

@app.route('/<int:year>/<model>/<int:odometer>/<condition>/<transmission>/<engine>/<cylinders>/<drive>')
def getPrice(year, model, odometer, condition, transmission, engine, cylinders, drive):
    # print(type(year), model, odometer, condition, transmission, engine, cylinders, drive)
    # Compute and return price
    price = getResult(year, model, odometer, condition, engine, transmission, cylinders, drive)
    return {'price': price}


if __name__ == "__main__":
    app.run()