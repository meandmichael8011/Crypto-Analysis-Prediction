IMPORTANT: This project is adapted for StreamLit, and thus is supposed to be run exclusively with StreamLit.

This pet project of mine includes WebScraping using Requests & BeautifulSoup4. The object of scraping is crypto & stock data from Yahoo Finance. As soon as the data is obtained, the script will move on to the next-day prediction, which involves the LSTM model fine-tuned specifically for this purpose of
scrutinizing the data (day and time) and coming up with a prediction for the next day. When the whole process is done, you will recieve a neat display of the current price per one currency unit, the price predicted for tomorrow and historical data. LSTM takes a little time to be trained, so 
don't be surprised that it can take you a minute or two. Also, since the project was done circa the New Year of 2024, I decided to add this effect of falling snowflakes for the sake of atmosphere.
Some fluctuations were observed when working with this prediction, so on no account should this project be considered any kind of financial advice. 
