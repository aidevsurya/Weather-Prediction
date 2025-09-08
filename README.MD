# Weather-ML-Model

<h1>🌦️ Seattle Weather Prediction using Machine Learning</h1>

<p>
This project uses <b>Random Forest Classifier</b> to predict weather conditions in Seattle based on features such as precipitation, temperature, and wind.<br>
It is trained on the <b>Seattle Weather Dataset</b> and achieves good accuracy in classifying weather types like <i>Rain, Sun, Snow, Fog, Drizzle</i>.
</p>

<hr>

<h2>✨ Features</h2>
<ul>
  <li>📊 Preprocessing with <b>Label Encoding</b></li>
  <li>🌲 Weather prediction using <b>Random Forest Classifier (Entropy criterion)</b></li>
  <li>💾 Model persistence with <b>Pickle</b></li>
  <li>🎯 Evaluation with accuracy score on test data</li>
  <li>🔮 Example prediction using custom input values</li>
</ul>

<hr>

<h2>🛠️ Requirements</h2>
<p>Make sure you have the following installed:</p>

<pre>
<code class="language-bash">
pip install pandas scikit-learn matplotlib
</code>
</pre>

<hr>

<h2>📂 Project Structure</h2>

<pre>
📁 Weather-Prediction
 ├── seattle-weather.csv      # Dataset
 ├── Weather_model.model      # Saved trained model (after running)
 ├── weather_predict.py       # Main script
 └── README.md                # Documentation
</pre>

<hr>

<h2>🚀 Usage</h2>

<ol>
  <li>Clone this repository:</li>
</ol>

<pre>
<code class="language-bash">
git clone https://github.com/aidevsurya/weather-prediction.git
cd weather-prediction
</code>
</pre>

<ol start="2">
  <li>Run the script:</li>
</ol>

<pre>
<code class="language-bash">
python weather.py
</code>
</pre>

<ol start="3">
  <li>Example output:</li>
</ol>

<pre>
Precipitation = 0.0
Max Temp = 8.3
Min Temp = 2.8
Wind = 4.1
Weather Prediction = Fog
Accuracy Score = 0.82
</pre>

<hr>

<h2>🔮 Prediction with Custom Values</h2>

<p>You can modify the input in the script to test your own values:</p>

<pre>
<code class="language-python">
Precipitation = 0.0
MaxTemp = 10.0
MinTemp = 3.0
Wind = 5.5
</code>
</pre>

<hr>

<h2>📊 Model Training Details</h2>
<ul>
  <li><b>Algorithm</b>: RandomForestClassifier (Entropy)</li>
  <li><b>Train/Test Split</b>: 80/20</li>
  <li><b>Encoded Labels</b>: Weather conditions converted using <code>LabelEncoder</code></li>
  <li><b>Evaluation Metric</b>: Accuracy</li>
</ul>

<hr>

<h2>📌 Future Improvements</h2>
<ul>
  <li>Add <b>visualizations</b> for predictions and feature importance</li>
  <li>Build a <b>Flask/Django web app</b> for interactive usage</li>
  <li>Experiment with other models (Logistic Regression, SVM, Gradient Boosting)</li>
</ul>

<hr>

<h2>📜 License</h2>
<p>This project is licensed under the <b>MIT License</b>.</p>
