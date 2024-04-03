# AGROBUDDY 🌿
#### A simple ML and DL based website which recommends the best crop to grow, fertilizers to use and the diseases caught by your crops.

## DISCLAIMER ⚠️
This is a POC(Proof of concept) kind-of project. The data used here comes up with no guarantee from the creator. So, don't use it for making farming decisions. If you do so, the creator is not responsible for anything. However, this project presents the idea that how we can use ML/DL into precision farming if developed at large scale and with authentic and verified data.

## MOTIVATION 💪
- Farming is one of the major sectors that influences a country’s economic growth. 

- In country like India, majority of the population is dependent on agriculture for their livelihood. Many new technologies, such as Machine Learning and Deep Learning, are being implemented into agriculture so that it is easier for farmers to grow and maximize their yield. 

- In this project, I present a website in which the following applications are implemented; Crop recommendation, Fertilizer recommendation and Plant disease prediction, respectively. 

    - In the crop recommendation application, the user can provide the soil data from their side and the application will predict which crop should the user grow. 
    
    - For the fertilizer recommendation application, the user can input the soil data and the type of crop they are growing, and the application will predict what the soil lacks or has excess of and will recommend improvements. 
    
    - For the last application, that is the plant disease prediction application, the user can input an image of a diseased plant leaf, and the application will predict what disease it is and will also give a little background about the disease and suggestions to cure it.

## DATA SOURCE 📊
- [Crop recommendation dataset ](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset) (custom built dataset)
- [Fertilizer suggestion dataset](https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/fertilizer.csv) (custom built dataset)
- [Disease detection dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)



# Built with 🛠️
<code><img height="30" src="data:image/png;base64,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"></code>
<code><img height="30" src="https://cdn.icon-icons.com/icons2/2415/PNG/512/java_original_logo_icon_146459.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/html/html.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/css/css.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/javascript/javascript.png"></code>
<code><img height="30" src="https://github.com/tomchen/stack-icons/raw/master/logos/bootstrap.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
<code><img height="30" src="https://symbols.getvecta.com/stencil_80/56_flask.3a79b5a056.jpg"></code>
<code><img height="30" src="https://cdn.iconscout.com/icon/free/png-256/heroku-225989.png"></code>

<code><img height="30" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="30" src="https://matplotlib.org/_static/logo2.svg"></code>
<code><img height="30" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pytorch/pytorch/39fa0b5d0a3b966a50dcd90b26e6c36942705d6d/docs/source/_static/img/pytorch-logo-dark.svg"></code>

## DEPLOYMENT 🚀

#### Deployment is done using [deploy](https://github.com/Gladiator07/Harvestify/tree/deploy) branch
#### This website is deployed at [Heroku](https://www.heroku.com/)
#### You can access it [here](https://harvestify.herokuapp.com/)
#### Note: The website may take a minute to load sometimes, as the server may be in hibernate state

## How to use 💻
- Crop Recommendation system ==> enter the corresponding nutrient values of your soil, state and city. Note that, the N-P-K (Nitrogen-Phosphorous-Pottasium) values to be entered should be the ratio between them. Refer [this website](https://www.gardeningknowhow.com/garden-how-to/soil-fertilizers/fertilizer-numbers-npk.htm) for more information.
Note: When you enter the city name, make sure to enter mostly common city names. Remote cities/towns may not be available in the [Weather API](https://openweathermap.org/) from where humidity, temperature data is fetched.

- Fertilizer suggestion system ==> Enter the nutrient contents of your soil and the crop you want to grow. The algorithm will tell which nutrient the soil has excess of or lacks. Accordingly, it will give suggestions for buying fertilizers.

- Disease Detection System ==> Upload an image of leaf of your plant. The algorithm will tell the crop type and whether it is diseased or healthy. If it is diseased, it will tell you the cause of the disease and suggest you how to prevent/cure the disease accordingly.
Note that, for now it only supports following crops

<details>
  <summary>Supported crops
</summary>

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Pepper
- Orange
- Peach
- Potato
- Soybean
- Strawberry
- Tomato
- Squash
- Raspberry
</details>

## How to run locally 🛠️
- Before the following steps make sure you have [git](https://git-scm.com/download), [Anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- Clone the complete project with `git clone https://github.com/Gladiator07/Harvestify.git` or you can just download the code and unzip it
- **Note:** The master branch doesn't have the updated code used for deployment, to download the updated code used for deployment you can use the following command
  ```
  ❯ git clone -b deploy https://github.com/Gladiator07/Harvestify.git 
  ```
- `deploy` branch has only the code required for deploying the app (rest of the code that was used for training the models, data preparation can be accessed on `master` branch)
- It is highly recommended to clone the deploy branch for running the project locally (the further steps apply only if you have the deploy branch cloned)
- Once the project is cloned, open anaconda prompt in the directory where the project was cloned and paste the following block
  ```
  conda create -n harvestify python=3.6.12
  pip install -r requirements.txt
  conda activate harvestify
  ```
- And finally run the project with
  ```
  python app.py
  ```
- Open the localhost url provided after running `app.py` and now you can use the project locally in your web browser.
## DEMO

- ### Crop recommendation system

![demo](https://media.giphy.com/media/90JbjdAa5nDq3TJh5u/giphy.gif)

- ### Fertilizer suggestion system

![demo](https://media.giphy.com/media/FLftUXMFo8N2bBjAXq/giphy.gif)


- ### Disease Detection system
![demo](https://media.giphy.com/media/NnMwEp2tGZdfnJbyjr/giphy.gif)



## Usage ⚙️
You can use this project for further developing it and adding your work in it. If you use this project, kindly mention the original source of the project and mention the link of this repo in your report.

## Further Improvements 📈
This was my first big project so there are lot of things to improve upon

- CSS code is totally messed up :pensive: (some code in file and some inline)
- Frontend can be made more nicer (PS: I suck at frontend development) :cry:	
- More data can be collected manually via web scrapping to make the system more accurate :monocle_face:	
- Additional plant images can be collected to make the disease detection part more robust and generalized :face_with_head_bandage:
- Modularized code can be written instead of writing in Jupyter Notebooks (will follow this in upcoming projects)


