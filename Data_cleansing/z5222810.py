import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


#Function to mapping countries names
def country_mapping(x):
    if re.search("Cabo Verde",str(x)):
        return "Cape Verde"
    elif re.search("Palestine",str(x)):
        return "Palestinian Territory"
    elif re.search("United States of America",str(x)):
        return "United States"
    elif re.search("Congo DR",str(x)):
        return "Democratic Republic of the Congo"
    elif re.search("Korea DPR",str(x)):
        return "North Korea"
    elif re.search("Lao PDR",str(x)):
        return "Laos"
    elif "Congo"==str(x).strip():
        return "Republic of the Congo"
    elif re.search("Brunei Daruss",str(x)):
        return "Brunei"
    elif re.search("Viet Nam",str(x)):
        return "Vietnam"
    elif re.search("Eswatini",str(x)):
        return "Swaziland"
    elif re.search("Ivoire",str(x)):
        return "Ivory Coast"
    elif re.search("Moldova Republic of",str(x)):
        return "Moldova"
    elif re.search("Russian Federation",str(x)):
        return "Russia"
    elif re.search("Korea Republic of",str(x)):
        return "South Korea"
    elif re.search("North Macedonia",str(x)):
        return "Macedonia"
    else:
        return x


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #Reading the exposure file
    df_exposure=pd.read_csv(exposure,sep=';',engine='python', encoding='utf-8',encoding_errors='ignore')
    
    #Drop null values of country column
    df_exposure.dropna(subset=['country'],axis=0,inplace=True)
    
    #Apply mapping function
    df_exposure['country']=df_exposure['country'].apply(country_mapping)
    
    #Reading Countries file
    df_countries=pd.read_csv(countries)
    
    df1=df_countries.merge(df_exposure, how='inner', left_on='Country', right_on='country')
    
    #drop 'country' column which is the same than 'Country' column
    df1.drop(['country'],axis=1,inplace=True)
    
    #Set 'Country' as index
    df1.set_index('Country',inplace=True)
    
    #Sort by index (ascending)
    df1.sort_index(axis=0,ascending=True,inplace=True)
    
    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


#This is to transform the column Cities in something possible to work  with
def trans_json(x):
    cities=x.split('|||')
    #Delete the duplicated information
    unique_cities=list(dict.fromkeys(cities))
    return [json.loads(unique_cities[i]) for i in range(len(unique_cities))]

#Function to get the average latitude and logitude per country
def avg_cal(x):
    latitude_list=[]
    longitude_list=[]
    for i in range(len(x)):
        latitude_list.append(x[i]['Latitude'])
        longitude_list.append(x[i]['Longitude'])
    latitude_avg=np.average(latitude_list)
    longitude_avg=np.average(longitude_list)
    return [latitude_avg,longitude_avg]

def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
        
    #transform the Cities column of df1 into disctionaries
    df1['Cities']= df1['Cities'].apply(trans_json)
    #Create a dataframe with average latitude and longitude for each country 
    aux_df=df1['Cities'].apply(avg_cal)
    
    #Separate latitude and longitud in different columns
    aux_df=pd.DataFrame(aux_df.to_list(),columns=['avg_latitude','avg_longitude'],index=aux_df.index)
    
    #the asked DataFrame
    df2=df1.join(aux_df)

    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2


#Function to get the distance from a point to Wuhan
def get_distance(x):
    lat_wuhan=math.radians(30.5928)
    long_wuhan=math.radians(114.3055)
    lat=math.radians(x[0])
    long=math.radians(x[1])
    dif_lat=lat-lat_wuhan
    dif_long=long-long_wuhan
    a= math.pow(math.sin(dif_lat/2),2)+math.cos(lat)*math.cos(lat_wuhan)*math.pow(math.sin(dif_long/2),2)
    distance= 2*6373*math.asin(math.sqrt(a))
    return distance

def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    dist_wuhan=df2[['avg_latitude','avg_longitude']].apply(get_distance,axis=1)
    #Transform serie to dataframe
    dist_wuhan=dist_wuhan.to_frame(name='distance_to_Wuhan')
    df3=df2.join(dist_wuhan).sort_values('distance_to_Wuhan',ascending=True)

    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3

#Continent mapping
def country_continent_mapping(x):
    if re.search("Burkina",str(x)):
        return "Burkina Faso"
    elif "CZ"==str(x).strip():
        return "Czech Republic"
    elif re.search("Congo, Democratic Repu",str(x)):
        return "Democratic Republic of the Congo"
    elif re.search("Burma",str(x)):
        return "Myanmar"
    elif re.search("Korea, North",str(x)):
        return "North Korea"
    elif "Congo"==str(x).strip():
        return "Republic of the Congo"
    elif re.search("Russian Federation",str(x)):
        return "Russia"
    elif re.search("Korea, South",str(x)):
        return "South Korea"
    elif "US"==str(x).strip():
        return "United States"
    else:
        return x

#Function to transform a string column with number representation into float numbers
def transform_index(x):
    try:
        x=float(x.replace(',','.'))
    except:
        x=float('NaN')
    return x

def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #read continent file
    df_continents=pd.read_csv(continents)
    #pre-processing to match countries between dataframes
    df_continents['Country']=df_continents['Country'].apply(country_continent_mapping)
    #mrge both dataframes
    aux_df=df2.merge(df_continents, how='inner', left_on='Country', right_on='Country')[['Continent','Country','Covid_19_Economic_exposure_index']]
    #apply transform_index to work with econnomic exposure index
    aux_df['Covid_19_Economic_exposure_index']=aux_df['Covid_19_Economic_exposure_index'].apply(transform_index)
    
    aux_df=aux_df[['Continent','Covid_19_Economic_exposure_index']]
    
    #The asked dataframe
    #df4=aux_df.astype({'Covid_19_Economic_exposure_index':'float64'}).groupby(by="Continent",as_index=True).mean().sort_values('Covid_19_Economic_exposure_index',ascending=True)
    
    df4=aux_df.groupby(by="Continent",as_index=True).mean().sort_values('Covid_19_Economic_exposure_index',ascending=True)


    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #Get just the asked columns
    df5_1=df2[['Income classification according to WB','Foreign direct investment','Net_ODA_received_perc_of_GNI']].copy()
    df5_1.rename(columns={'Income classification according to WB':'Income Class'},inplace=True)
    
    #Apply transformation into the index to work with them
    df5_1['Foreign direct investment']=df5_1['Foreign direct investment'].apply(transform_index)
    df5_1['Net_ODA_received_perc_of_GNI']=df5_1['Net_ODA_received_perc_of_GNI'].apply(transform_index)
    
    #get the mean by Income Class
    df5=df5_1.groupby(by='Income Class').mean()
    
    #rename the columns
    df5.rename(columns={'Foreign direct investment':'Avg Foreign direct investment','Net_ODA_received_perc_of_GNI':'Avg_Net_ODA_received_perc_of_GNI'},inplace=True)
    

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


#Function to get the population of every city
def city_population(x): 
    cities_name_list=[]
    population_list=[]
    for i in range(len(x)):
        if x[i]['Population']:
            cities_name_list.append(x[i]['City'])
            population_list.append(x[i]['Population'])

    return pd.Series({'City_name':cities_name_list,'City_population':population_list})

def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    cities_lst = []
    #Selecting just the useful columns
    df6_1=df2[['Cities','Income classification according to WB']].copy()
    #Filter the countries
    df6_1.query('`Income classification according to WB`=="LIC"',inplace=True)
    
    #Apply city_population function to the dataframe
    df6_1=df6_1['Cities'].apply(city_population).apply(pd.Series.explode,axis=0)
    
    lst=list(df6_1.sort_values('City_population',ascending=False)['City_name'].iloc[:5])
    cities_lst=lst.copy()
    log("QUESTION 6", output_df=None, other=cities_lst)
    return lst


def cities_names(x):
    #for i in range(len(x)):
    #    x[i]['City']
    aux=[re.sub('`','',x[i]['City']) for i in range(len(x))]
    return list(dict.fromkeys(aux))

def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    
    #Get just the needed columns
    df7_1=df2[['Cities']].copy()
    #transform cities column
    #df7_1=df7_1['Cities'].apply(trans_json)
    #get cities name
    df7_1=df7_1['Cities'].apply(cities_names).explode() 
    #change some column names
    df7_1=df7_1.reset_index().rename(columns={'Country':'countries','Cities':'city'})
    
    #Filter cities that appear in more than one country
    df7_2=df7_1.groupby(by='city').count().query('countries > 1')
    #change some column names
    df7_2=df7_2.reset_index().rename(columns={'countries':'numb_countries'})
    
    #merge dataframes
    df7=df7_2.merge(df7_1,how='left',left_on='city',right_on='city')
    #get the asked dataframe
    df7=df7.groupby(['city']).agg({'countries':lambda x: list(x)})
    

    log("QUESTION 7", output_df=df7, other=df7.shape)
    return df7

#Function to get countries population
def contries_pop(x):
    total_country_pop=0
    for i in range(len(x)):
        if x[i]['Population']:
            total_country_pop= total_country_pop + x[i]['Population']
    return total_country_pop

def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """
    
    df_continents=pd.read_csv(continents)
    
    #population per country
    pop_per_country=df2['Cities'].apply(contries_pop)
    
    #world population
    world_pop=pop_per_country.sum()
    
    #Getting the south american countries
    south_amer_countries=pd.DataFrame(df_continents.query('Continent=="South America"')['Country'])
    
    #Getting the population per each south american country
    south_ame_pop=south_amer_countries.merge(pop_per_country,how='left',left_on='Country',right_on='Country').rename(columns={'Cities':'Population'})
    
    #population expresses in percentage
    south_ame_pop['Population']=south_ame_pop['Population'].apply(lambda x: (x/world_pop)*100)
    south_ame_pop=south_ame_pop.rename(columns={'Population':'Population%'}).sort_values('Population%')
    
    #Making the plot 
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10,8))
    
    countries = south_ame_pop['Country']
    y_pos = np.arange(len(countries))
    population = south_ame_pop['Population%']
    
    ax.barh(y_pos, population)
    ax.set_xlim(0,6.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries)
    
    ax.set_xlabel('Population Percentage (%)',fontsize=14)
    ax.set_ylabel('Countries',fontsize=14)
    #ax.legend(loc='center right',fontsize=12)
    ax.set_title('Percentage of the world population in South American countries',fontsize=18)
    
    for i in ax.patches:
        ax.text(i.get_width()+0.05, i.get_y()+0.15,str(round(i.get_width(), 3))+'%', fontsize=12,color='dimgrey')
    
    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """
    df9=df2.copy()
    #pre=processing: transform strings in floats
    df9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI']=df9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'].apply(transform_index)
    df9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import']=df9['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'].apply(transform_index)
    df9['Foreign direct investment, net inflows percent of GDP']=df9['Foreign direct investment, net inflows percent of GDP'].apply(transform_index)
    df9['Foreign direct investment']=df9['Foreign direct investment'].apply(transform_index)
    
    #Get the needed columns
    df9_1=df9[['Income classification according to WB','Covid_19_Economic_exposure_index_Ex_aid_and_FDI',\
               'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',\
                   'Foreign direct investment, net inflows percent of GDP','Foreign direct investment']].copy()
    
    df9_1=df9_1.groupby(by='Income classification according to WB').mean().reindex(index=['LIC','MIC','HIC'])
    
    #Making the plot
    fig, axs = plt.subplots(2, 2, figsize=(14,11))
    fig.suptitle('Metrics VS different Countries income level',fontsize=16)
    axs[0,0].set_title('Covid_19_Economic_exposure_index_Ex_aid_and_FDI',fontsize=10)
    axs[0,0].bar(df9_1.index, df9_1['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'], color=['purple','green','blue'])
    axs[0,0].set_ylim(3.5,4)
    axs[0,0].set_ylabel('Index Average',fontsize=9)
    axs[0,0].set_xlabel('Income Level',fontsize=9)
    
    axs[0,1].set_title('Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',fontsize=10)
    axs[0,1].bar(df9_1.index, df9_1['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'], color=['purple','green','blue'])
    axs[0,1].set_ylim(3.4,3.8)
    axs[0,1].set_ylabel('Index Average',fontsize=9)
    axs[0,1].set_xlabel('Income Level',fontsize=9)
    
    axs[1,0].set_title('Foreign direct investment, net inflows percent of GDP',fontsize=10)
    axs[1,0].bar(df9_1.index, df9_1['Foreign direct investment, net inflows percent of GDP'], color=['purple','green','blue'])
    axs[1,0].set_ylim(1.5,5.5)
    axs[1,0].set_ylabel('Average %',fontsize=9)
    axs[1,0].set_xlabel('Income Level',fontsize=9)
    
    axs[1,1].set_title('Foreign direct investment',fontsize=10)
    axs[1,1].bar(df9_1.index, df9_1['Foreign direct investment'], color=['purple','green','blue'])
    axs[1,1].set_ylim(1,2)
    axs[1,1].set_ylabel('Index Average',fontsize=9)
    axs[1,1].set_xlabel('Income Level',fontsize=9)

    plt.savefig("{}-Q12.png".format(studentid))


def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """
    #read continent file
    df_continents=pd.read_csv('Countries-Continents.csv')
    
    #pre-processing to match countries between dataframes
    df_continents['Country']=df_continents['Country'].apply(country_continent_mapping)
    
    #mrge both dataframes
    df10=df2.merge(df_continents, how='inner', left_on='Country', right_on='Country')
    df10['Cities']=df10['Cities'].apply(contries_pop)
    df10=df10.rename(columns={'Cities':'Population'})
    df10_1=df10[['Country','Population','Continent','avg_latitude','avg_longitude']].copy()
    total_pop=df10_1['Population'].sum()
    
    #make the population column a percentage (to make the relative size for the plot)
    df10_1['Population']=df10_1['Population'].apply(lambda x: (x/total_pop)*5000)
    
    #make the plot
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15,8))
    list_colours=['brown','red','green','pink','blue','orange']
    #conti_array= df_continents['Continent'].unique()
    conti_array=['Asia','North America','South America','Europe','Africa','Oceania']
    continent_colours=zip(conti_array,list_colours)
    for i,j in continent_colours:
        df_conti= df10_1.query('Continent==@i')
        x_lon=df_conti['avg_longitude']
        y_lat=df_conti['avg_latitude']
        plt.scatter(x_lon, y_lat, s=df_conti['Population'], c=j ,label=df_conti['Country'])
    
    ax.set_ylim(-45,70)
    ax.set_xlim(-190,190)
    ax.set_xlabel('Longitud',fontsize=14)
    ax.set_ylabel('Latitud',fontsize=14)
    ax.set_title('Location of Countries and Size Population Comparison',fontsize=18)
    plt.legend(conti_array,loc='upper left',title='Continents',title_fontsize='large',fontsize='medium',labelspacing=1,\
              markerscale=1)
    

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
     
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv") 
    df5 = question_5(df2.copy(True))
    lst = question_6(df2.copy(True))
    df7 = question_7(df2.copy(True))
    question_8(df2.copy(True), "Countries-Continents.csv")
    question_9(df2.copy(True))
    question_10(df2.copy(True), "Countries-Continents.csv")
  