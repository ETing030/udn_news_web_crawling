# udn_news_web_crawling
大數據分析與資料探勘期末小組報告部分內容:D  
部分程式說明仍未補上，預計 4/25 前完成

## 關於 此期末小組報告
目的：利用機器學習，訓練出一個可以將新聞內容自動歸類的模型  
大致步驟：
1. 透過爬蟲蒐集資料，也就是新聞資訊，包含其標題、內文、標籤／類別等。新聞來源：聯合報 4000 筆、ETtoday 新聞雲 4100 筆及風傳媒 1140 筆，共 9240 筆新聞資料
2. 對新聞內文進行刪去英數字和斷詞處理，並統整所有文章的類別，將類別限縮到 15 種
3. 利用 Term Frequency 與 TF-IDF 進行特徵處理
4. 用不同機器學習模型，以 80% 的資料為訓練集訓練，途中嘗試不同的參數值。模型：Logistic Regression, SVM, XGBoost, Random Forest
5. 以剩下 20% 的資料來測試模型，並觀察其分類結果  

註：特徵處理與訓練模型的部分是來自  
https://tlyu0419.github.io/2020/04/04/Text-Classification/?fbclid=IwAR2t27jNAu-FxEQs0r2UwAmQZu_HEH0Ou9aYibQJAnvh9L9gZk1dTMcI0Fc  
但目前連結無效，不過 github 找的到人 :)

## 關於 此 repo
為報告中處理聯合報資料的部分，不包含限縮（統整）類別，且針對聯合報的資料單獨進行特徵處理與訓練模型  
註：此處進行之特徵處理與訓練模型僅作流程參考，並不涵蓋於期末報告中，因資料仍需涵蓋 ETtoday 和 風傳媒 之資料再一同分析  
註：聯合報的網頁稱為聯合新聞網，後續會以此稱之

### `聯合新聞網爬蟲.ipynb`
- 針對聯合新聞網即時不分類新聞（
https://udn.com/news/breaknews/1/99#breaknews
）進行爬取，由於網頁為 infinity scroll （無限滾軸），因此先用 Selenium 套件進行動態爬蟲，將頁面滑至所要求**頁數**，再透過 BeautifulSoup 套件，把所要的資訊，包含網頁標題、內文、連結和標籤／分類等資訊，節錄下來，以 Dataframe 的形式，存於檔案中，並匯出資料（`聯合報.csv`）
- 對於上點**頁數**之解釋，滾軸式網頁為將一頁頁的資料，不以網址之次連結分別，而是全放在同一個網址下，當網頁下拉至一定高度時，便會載入下一頁之資料，因此，仍可用頁數來衡量要抓取的資料量。在聯合新聞網中，一頁有 20 則新聞，以下為程式中執行下拉至特定頁數的區塊
<p align=center>
  <img src='https://user-images.githubusercontent.com/39528069/163224341-fbe6959a-7301-46f9-9177-da792c4ca646.png' width='900'>
</p>

- 在使用 Selenium 套件中，會使用到以 Chrome 作為瀏覽器之 Webdriver，即 `ChromeDriver.exe`， 且 ChromeDriver 之版本需與執行程式之電腦，其 Chrome 版本相符，如在上次測試`聯合新聞網爬蟲.ipynb`時瀏覽器版本為 98.0.4758.102，因此用版本同為 98.0.4758.102 的 `chromedriver.exe`（附於此 repo 中）來進行  
其他版本之 ChromeDriver 執行檔可由此網址下載：https://chromedriver.storage.googleapis.com/index.html  
附上在`聯合新聞網爬蟲.ipynb`針對版本變更的說明
<p align=center>
  <img src='https://user-images.githubusercontent.com/39528069/163222677-16ca3e4d-115f-4cc0-97c0-5fb66a8ef2f7.png' width='900'>
</p>

- 從 BeautifulSoup 抓取原始資料後，收集我們所需要的片段，下圖為所抓取之原始資料與新聞界面對應
<p align=center>
  <img src='https://user-images.githubusercontent.com/39528069/163224622-91db816a-9601-4ac0-baeb-4d0468730252.png' width='900'>
</p>  
　　下圖為整理過後的資料
<p align=center>
  <img src='https://user-images.githubusercontent.com/39528069/163222236-21090b8f-6ba3-4a13-b42d-3a8a973b70da.png' width='900'>
</p>


### `聯合新聞網資料處理與分類.ipynb`
- 針對 `聯合報.csv` 進行內文處理、統整類別、特徵處理並測試訓練模型
- `聯合報.csv` 為當初作為訓練模型的資料，共 4000 筆
- 由於聯合新聞網的部分新聞連結會轉移至其他頁面格式（要抓取之內容所對應的 xpath 與他者不同），或是該連結網站須為 VIP 才能閱覽的文章，使得這些新聞可能沒有標籤或內文，必須將其除去。之後，以**人工的方式**將新聞中非內文的部分除去，再將內文轉換為 json 套件可以讀取的形式，並用 re 套件捨去標點符號、英文及數字。統整類別的部分則是只留下主類別，亦即 tag 欄位中第一個詞，最後再用 jieba 套件將內文斷詞
- 針對有無斷詞的內文分別以 sklearn.feature_extraction.text 中的 CountVectorizer 和 TfidfVectorizer 進行特徵處理，並以 LogisticRegression 作為模型，用 80% 的資料去訓練，並測試其準確度

<div align=center>
 
  內文  | 特徵處理 | training accuracy | testing accuracy 
:---:  | :---: | :---: | :---: 
無斷詞  | CountVectorizer | 0.9671 | 0.7434
無斷詞  | TfidfVectorizer | 0.8470 | 0.6814
有斷詞  | CountVectorizer | 0.9984 | 0.8420
有斷詞  | TfidfVectorizer | 0.9074 | 0.7826
  
</div>

### `新聞內容統整和文字雲.ipynb`
- 針對 `新聞分類統整.csv` 進行內文處理、統整類別、特徵處理並測試訓練模型
- `新聞分類統整.csv` 即為起初所說，約 9240 筆資料，經前面新聞聯合網刪去部分文章後，剩下約 9200 筆資料
- 同樣地，
- 
