#import modules
from os  import system
import  joblib  as  jb
import  pandas  as  pd
import  numpy  as  np
from  sklearn.linear_model  import  LinearRegression

#filter data
data = pd.read_csv("Salary.csv")
#print(data.head(5))
#print(data.shape)

X = data["YearsExperience"]
y = data["Salary"]

X = X.values.reshape(30,1)

#model creation
mind = LinearRegression() 
mind.fit(X, y)


while True:
	print("-------------------------------------------------------------------------------------")
	print("\t\t\t *********Salary Predictor Program********* ")

	exp = float(input("\nEnter your Experience:  "))
	print("Your estimated Salary:", mind.predict( [[ exp ]] ) )
	print("-------------------------------------------------------------------------------------")

	quit = input("\nPress q to exit & to continue press Enter:  ")
	print()
	if quit == "q":
	 	break

jb.dump( mind, "salary.pkl")
	


