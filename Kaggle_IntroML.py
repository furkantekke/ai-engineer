import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dataset_path = '../data.csv' 
dataset = pd.read_csv(dataset_path)

dataset.describe() #data kolonları hakkında count,mean,std, min, kolon verileri sıralandığında ne kadarı %25-%50-%75'inden büyük olur, max bilgileri verir

print(dataset.columns) #datasette geçen kolonları alabilirsin

y=dataset.column1 #datasette column1 olarak mevcut kolon değerlerini y dataframe'ine atar
 
feature_names =["column1","column2"] #eğer sadece 2 kolonu değerlendirmeye almak istersek kolon isimlerini yazarak 

X = dataset[feature_names] #datasetten sadece seçtiğimiz kolonları bir dataframe'e almak istersek

furkan_model = DecisionTreeRegressor(random_state=1) #random_state=1 her çalıştırıldığında aynı sayı gelmesi için sabit bir değerdir.
furkan_model.fit(X, y) #model X kümesi yani seçtiğimiz kolonlara bakarak Y dataframe'ini öğrenmeye çalışır.Yani bunlar arasındaki ilişkiyi çıkartır ve modelin kendisini günceller. Yeni bir değişkene atanmasına gerek yoktur.

tahminler = furkan_model.predict(X) #seçilen kolonlar bazında prediction değerler üretip tahminler dataframe'ine aktarır.

mean_absolute_error(y, tahminler) #MAE tahminlerin gerçek değerler ile ne kadar yaklaşık olduğunu bulmak için bir hesaplamadır.

------------------------------------------------------------------------
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)#en doğru yöntem veri setini eğitim ve test için bölmektir.
furkan_model2 = DecisionTreeRegressor() 
furkan_model2.fit(train_X, train_y) #train başlangıçlı dataset ve sonuçlar tamamen eğitimde kullanılmak üzere
tahminler2 = furkan_model2.predict(val_X) #val başlangıçlı dataset ve sonuçlar eğitilen modelin testinde kullanılmak üzere
tahminler2_mae = mean_absolute_error(val_y, tahminler2) #gerçek sonuç değerleri ile tahmin edilen sonuçlar arasondaki mae hesabı yapılır.
-------------------------------------------------------------------------

 furkan_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1) #max_leaf_nodes vermek decision tree algoritmasının overfitting yapmasını önler ve yaprak sayısını sınırlar.Decision tree algoritmasında node sayısı dataset'te ilişkilerin kaç parçada inceleneceğini belirler. node sayısını çok arttırmak overfitting yani datanın ezberlenmesine, çok düşük tutmak da underfitting dataset ilişkilerinin tam anlaşılamamasına ve düşük doğrulukta değerler üretilmesine neden olur.
#optimum node değeri aşağıdaki gibi bir for döngüsünde mae'lerin karşılaştırılması yoluyla bulunur.Kod örneği aşağıdadır.

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

-----------------------------------------------------------------------

-------------- RANDOM FORESTS ---------------