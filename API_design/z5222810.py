#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pablo Pacheco
znumber= z5222810
"""

import requests
import json
import pandas as pd
from flask import Flask
from flask import request, send_file
from flask_restx import Resource, Api
from flask_restx import fields
from flask_restx import reqparse
import datetime
import sqlite3
from sqlite3 import Error
from pandas.io import sql
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
api = Api(app, default='TV Shows',title='TV Shows database',\
          description='This is an app to get information about different TV shows.')

parser = reqparse.RequestParser()
parser.add_argument('name', type=str)    
 
schedule_model=api.model("attributes_schedule",{
    "time": fields.String(description='The format is HH:MM',pattern='[0-2][0-9]:[0-5][0-9]'),
    "days":fields.List(fields.String)})
rating_model=api.model("attributes_rating",{"average":fields.Float})
country_model=api.model("attributes_country",{
    "name": fields.String,
    "code": fields.String,
    "timezone": fields.String})
network_model=api.model("attributes_network",{
    "id": fields.Integer,
    "name": fields.String,
    "country": fields.Nested(country_model)
    })
# The schema of a tv show - id, tvmaze-id and last-update are not included because the user is not permitted to modify those attributes
tvshow_model = api.model('tvshow', {
    'name': fields.String,
    'type': fields.String,
    'language': fields.String,
    'genres': fields.List(fields.String),
    'status': fields.String,
    'runtime': fields.Integer,
    'premiered': fields.Date(description="the format is yyyy-mm-dd"),
    'officialSite': fields.String,
    'schedule': fields.Nested(schedule_model),
    'rating':fields.Nested(rating_model),
    'weight': fields.Integer,
    'network': fields.Nested(network_model),
    'summary': fields.String
    
})

#Create a database
def db_creation(db_name):
    try:    
        cnx = sqlite3.connect(db_name)
        
    except Error:
        print(Error)
        
    finally:
        cnx.close()

#write dataframe into sql db
def db_write(df, database_file, table_name):
    cnx = sqlite3.connect(database_file)
    df.to_sql(table_name, cnx, if_exists='append', index=False)
    cnx.close()

#Create table in sql database
def create_table(database_name, create_table_sql):
    cnx = sqlite3.connect(database_name)
    try:
        cursorObj = cnx.cursor()
        cursorObj.execute(create_table_sql)
    except Error:
        print(Error)
    cnx.close()

#read sql table and return a dataframe
def read_from_sqlite(database_file, table_name):
    cnx = sqlite3.connect(database_file)
    df_aux=sql.read_sql('select * from ' + table_name, cnx)
    cnx.close()
    return df_aux

#create the dictionary links
def links_creation(host_name,port_number,id,current_index, df_aux):
    self_link='http://'+host_name+':'+str(port_number)+'/tv-shows/'+str(id)
    links_dict={"self":{"href":self_link}}
    #check if there is a previous resource
    if current_index-1 in df_aux.index:
        prev_id=int(df_aux.loc[current_index-1,'id'])
        prev_link='http://'+host_name+':'+str(port_number)+'/tv-shows/'+str(prev_id)
        links_dict["previous"]={"href":prev_link}
        
    if current_index+1 in df_aux.index:
        next_id=int(df_aux.loc[current_index+1,'id'])
        next_link='http://'+host_name+':'+str(port_number)+'/tv-shows/'+str(next_id)
        links_dict["next"]={"href":next_link}
        
    return links_dict

   
#Question1
@api.route('/tv-shows/import')
@api.param('name', 'Title for the tv show')
class TvShowImport(Resource):
    @api.response(201,'Created')
    @api.response(200,'OK')
    @api.response(404,'Not Found')
    @api.doc(description='Import TV show to the application')
    def post(self):
        args = parser.parse_args()
        r=requests.get("http://api.tvmaze.com/search/shows", params={'q':args.get('name')})
        show=r.json()
        if not show:
            api.abort(404, "TV show was not found")
            
        #get just the first result in case multiple answers
        show=show[0]
        id_tvmaze=show['show']['id']
       
        #check if the requested tv-show is in the own database already
        df_aux=read_from_sqlite(datab_name, tab_name)
        if id_tvmaze in df_aux['tvmaze_id'].values:     
            return {"message":"TV-show is already in the database"},200
        
        #keep important information of the TV show
        last_up= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if df_aux['id'].empty:
            new_id=1
            #links dictionary
            link='http://'+host_name+':'+str(port_number)+'/tv-shows/'+str(new_id)
            links_dict={"self":{"href":link}}
        else:
            new_id=int(df_aux['id'].max()+1)
            #links dictionary
            link='http://'+host_name+':'+str(port_number)+'/tv-shows/'+str(new_id)
            link_prev='http://'+host_name+':'+str(port_number)+'/tv-shows/'+str(df_aux['id'].max())
            links_dict={"self":{"href":link},"previous":{"href":link_prev}}
        
        tv_name=show['show']['name']
        
        important_attributes=['name','type','language','genres','status','runtime','premiered','officialSite',\
                              'schedule','rating','weight','network','summary']
        
        import_attr_dict={'id':[new_id],'last_update':last_up,'tvmaze_id':id_tvmaze}
        for i in important_attributes:
            if isinstance(show['show'][i],dict):
                import_attr_dict[i]=json.dumps(show['show'][i])
            elif isinstance(show['show'][i],list):
                import_attr_dict[i]=json.dumps({'str_json':show['show'][i]})
            else:
                import_attr_dict[i]=show['show'][i]
        
       
        df_request=pd.DataFrame(import_attr_dict)
        
        db_write(df_request, datab_name, tab_name)
        
        
        dict_to_return={'id':int(new_id),'last-update':last_up,'tvmaze-id':int(id_tvmaze),'name':tv_name,\
                   '_links':links_dict}
        
        return dict_to_return,201


#Question2
@api.route('/tv-shows/<int:id>')
@api.param('id', 'The tv-show identifier')
class TvShowRetrieve(Resource):
    @api.response(200,'OK')
    @api.response(404,'Not Found')
    @api.doc(description='Retrieve a TV show information from the application')
    def get(self,id):
        #check if the requested tv-show is in the database
        df_aux=read_from_sqlite(datab_name, tab_name)
        if id not in df_aux['id'].values:
            api.abort(404, "TV show {} was not found".format(id))
        
        current_index=int(df_aux['id'][df_aux['id']==id].index[0])
        dict_to_return={}
        for i in df_aux.columns:
            if i=='genres':
                dict_to_return[i]=json.loads(df_aux.loc[current_index,i])['str_json']
            elif i in ['schedule','rating','network']:
                dict_to_return[i]=json.loads(df_aux.loc[current_index,i])
            elif i in ['id','tvmaze_id','runtime','weight']:
                if i =='tvmaze_id':
                    dict_to_return['tvmaze-id']=int(df_aux.loc[current_index,i])
                else:
                    dict_to_return[i]=int(df_aux.loc[current_index,i])
            else:
                if i=='last_update':
                    dict_to_return['last-update']=df_aux.loc[current_index,i]
                else:
                    dict_to_return[i]=df_aux.loc[current_index,i]
                
        
        dict_to_return["_links"]= links_creation(host_name,port_number,id,current_index, df_aux)        
            
        return dict_to_return,200



#Question 3
    @api.response(404, "TV show was not found")
    @api.response(200, 'OK')
    @api.doc(description="Delete a tv show by its ID")
    def delete(self,id):
        #check if the requested tv-show is in the database
        df_aux=read_from_sqlite(datab_name, tab_name)
        if id not in df_aux['id'].values:
            api.abort(404, "The TV show with id {} couldn't be deleted because it was not found".format(id))

        cnx=sqlite3.connect(datab_name)
        cursorObj=cnx.cursor()
        query='DELETE FROM '+tab_name+' where id='+str(id)
        cursorObj.execute(query)
        cnx.commit()
        cnx.close()
        
        return {"message" :"The tv show with id {} was removed from the database!".format(id),\
                "id": id},200


#Question4
    @api.response(404, 'TV show was not found')
    @api.response(400, 'Validation Error')
    @api.response(200, 'OK')
    @api.expect(tvshow_model, validate=True)
    @api.doc(description="Update a tv show by its ID")
    def patch(self,id):
        #check if the requested tv-show is in the database
        df_aux=read_from_sqlite(datab_name, tab_name)
        if id not in df_aux['id'].values:
            api.abort(404, "The TV show with id {} couldn't be updated because it was not found".format(id))

        #convert the request to json
        tvshow_req=request.json
                
        id_index=df_aux['id'][df_aux['id']==id].index[0]

        #Check the keys
        for k in tvshow_req:
            if k not in tvshow_model.keys():
                return {"message": "Attribute {} is invalid".format(k)}, 400
            if k == 'schedule':
                for i in tvshow_req[k]:
                    if i not in ['time','days']:
                        return {"message": "Attribute {} is invalid".format(i)}, 400
            if k == 'rating':
                for i in tvshow_req[k]:
                    if i not in ['average']:
                        return {"message": "Attribute {} is invalid".format(i)}, 400                    
            if k == 'network':
                for i in tvshow_req[k]:
                    if i not in ['id','name','country']:
                        return {"message": "Attribute {} is invalid".format(i)}, 400
                    if i == 'country':
                        for j in tvshow_req[k][i]:
                            if j not in ['name','code','timezone']:
                                return {"message": "Attribute {} is invalid".format(j)}, 400
            
        
        #open db conexion
        cnx=sqlite3.connect(datab_name)    
        
        #Update a value just if it is different           
        for k in tvshow_req:
            value_req=tvshow_req[k]
            if isinstance(value_req,dict):
                db_dict=json.loads(df_aux.loc[id_index,k])
                if k =='network':
                    for i in value_req.keys():
                        if i in ['id','name']:
                            db_dict[i] = value_req[i]
                        else:
                            for j in value_req['country'].keys():
                                db_dict['country'][j]=value_req['country'][j]
                                
                else:
                    for i in value_req.keys():
                        db_dict[i] = value_req[i] 
                
                value_req=json.dumps(db_dict)
                            
            elif isinstance(value_req,list):
                value_req=json.dumps({'str_json':value_req})
           
            
            if value_req != df_aux.loc[id_index,k]:
                cursorObj=cnx.cursor()
                updated_value=str(value_req)
                query=f'UPDATE {tab_name} SET {k} = ? WHERE id = {id}'
                cursorObj.execute(query,(updated_value,))
                cnx.commit()
        
        last_up= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursorObj=cnx.cursor()
        query=f'UPDATE {tab_name} SET last_update = ? WHERE id = {id}'
        cursorObj.execute(query,(last_up,))
        cnx.commit()
        
        cnx.close()
           
        
        dict_to_return={'id':id, 'last-update':last_up}
        
        #create the dictionary links 
        dict_to_return["_links"]= links_creation(host_name,port_number,id,id_index, df_aux)
               
        return dict_to_return,200


#Question 5

order_by_help='''Put + to ascending, - to descending, in front of every attribute. More than 
                one attribute could be given (separated by commas: CSV-format), the first 
                ones are going to have the sorting preference. These are the only attributes 
                which can be used: [id, name, runtime, premiered, rating-average]'''
                        
filter_help='''It determines which attributes should be shown for each tv show. It is a CSV-format
                value (it can be applied more than one attribute at the same time) and just can be 
                considered the next attributes: [tvmaze_id ,id ,last-update ,name ,type ,language,
                                                 genres ,status ,runtime ,premiered ,officialSite,
                                                 schedule ,rating ,weight ,network ,summary] '''

parser_2 = reqparse.RequestParser()
parser_2.add_argument('order_by',type=str,default='+id',help=order_by_help)
parser_2.add_argument('page',type=int,default=1)
parser_2.add_argument('page_size',type=int,default=100)
parser_2.add_argument('filter',type=str,default='id,name',help=filter_help)


@api.route('/tv-shows')
class TVshowsList(Resource):

    @api.response(404, 'TV shows were not found')
    @api.response(400, 'Validation Error')
    @api.response(200, 'OK')
    @api.doc(description="Retrieve the list of available TV shows")
    @api.expect(parser_2)
    def get(self):
        args = parser_2.parse_args()

        # retrieve the query parameters
        order_by = args.get('order_by')
        page= args.get('page')
        page_size= args.get('page_size')
        filter_att= args.get('filter')
        
        #Check if page and page are permitted
        if page <=0 or page_size<=0:
            return {"message":"page and page_size should be integers greater than 0"},400
        
        
        #Convert the db table in a dataframe
        df_aux=read_from_sqlite(datab_name, tab_name)
        #Rename columns with _ to -
        df_aux.rename(columns={'tvmaze_id':'tvmaze-id','last_update': 'last-update'}, inplace=True)
        
        #Check if the db table was empty
        if df_aux.empty:
            return {"message":"There are no tv shows to retrieve"},404
                
        #Clean the order_by input
        order_by=order_by.replace(' ','')
        if order_by[-1]==',':
            order_by=order_by[:-1]
        
        #sorting according to order_by
        list_order=[]
        list_ascending=[]
        for i in order_by.split(','):
            if i[1:] not in ['id','name','premiered','rating-average']:
                return {"message":"{} is an invalid attribute for order_by".format(i[1:])},400
            list_order.append(i[1:])
            if i[0] not in ['-','+']:
                return {"message":"in front of every attribute is expected a + or -"},400
            if i[0]=='+':
                list_ascending.append(True)
            else:
                list_ascending.append(False)
        
        #Adding rating-average column to the dataframe if it was given in the params
        if 'rating-average' in list_order:
            df_aux['rating-average']=df_aux['rating'].apply(lambda x: json.loads(x)['average'])
         
        df_aux.sort_values(by=list_order,ascending=list_ascending,inplace=True)
      
        #Clean the filter input
        filter_att=filter_att.replace(' ','')
        if filter_att[-1]==',':
            filter_att=filter_att[:-1]
            
        #Apply filters
        columns_to_keep=filter_att.split(',')        
        #check if the filter values are permitted values
        permitted_filters=['tvmaze-id','id' ,'last-update' ,'name' ,'type' ,'language' ,'genres' ,'status' ,'runtime' ,'premiered' ,'officialSite' ,'schedule' ,'rating' ,'weight' ,'network' ,'summary']
        for i in columns_to_keep:
            if i not in permitted_filters:
                return {"message": "Attribute {} is invalid".format(i)}, 400
        #Check if there is repeated values
        rep_values=set()
        for i in columns_to_keep:
            if i not in rep_values:
                rep_values.add(i)
            else:
                return {"message": "An attribute cannot appeared twice in the filter"},400
        
        #create the dataframe with the asked columns
        df_aux=df_aux[columns_to_keep]
        
        #Calculate the max number of pages
        max_number_pages=df_aux.shape[0]/page_size
        if max_number_pages%1 !=0:
            max_number_pages=int(max_number_pages + 1)
        
        max_number_pages=int(max_number_pages)
        
        if page > max_number_pages:
            return {"message":"The page number cannot be greater than {}".format(max_number_pages)},400
        
        #Create the dataframe for the given page number and links dict
        links_dict=dict()
        self_link= f'http://{host_name}:{port_number}/tv-shows?order_by={order_by}&page={page}&page_size={page_size}&filter={filter_att}'
        prev_link= f'http://{host_name}:{port_number}/tv-shows?order_by={order_by}&page={page-1}&page_size={page_size}&filter={filter_att}'
        next_link= f'http://{host_name}:{port_number}/tv-shows?order_by={order_by}&page={page+1}&page_size={page_size}&filter={filter_att}'
        
        if page==1:
            from_row=0
            until_row=page_size
            links_dict["self"]={'href':self_link}
            if max_number_pages != 1:
                links_dict['next']={'href':next_link}
        elif page==max_number_pages:
            from_row=(max_number_pages - 1)*page_size
            until_row=max_number_pages+1
            links_dict["self"]={'href':self_link}
            links_dict["previous"]={'href':prev_link}
        else:
            from_row=(page - 1)*page_size
            until_row= (page - 1)*page_size + page_size
            links_dict["self"]={'href':self_link}
            links_dict["previous"]={'href':prev_link}
            links_dict['next']={'href':next_link}
            
        df_aux=df_aux[from_row:until_row]
                
        #Transform the dataframe in a list of jsons
        list_of_shows=json.loads(df_aux.to_json(orient='records'))
        for i in range(len(list_of_shows)):
            for k in list_of_shows[i].keys(): 
                if k =='genres':
                    list_of_shows[i][k]=json.loads(list_of_shows[i][k])['str_json']
                    #print(json.loads(list_of_shows[i][k])['str_json'])
                elif k in ['schedule','rating','network'] and list_of_shows[i][k] is not None:
                    list_of_shows[i][k]=json.loads(list_of_shows[i][k])
        
             
        dict_to_return={'page':page,'page-size':page_size, 'tv-shows':list_of_shows,'_links':links_dict}
        #return json.loads(df_aux.to_json(orient='records'))
        return dict_to_return,200


#Question 6
parser_3 = reqparse.RequestParser()
parser_3.add_argument('format',type=str,default='image',choices=['json','image'],help='Choose answer format (json or image)')
parser_3.add_argument('by',type=str,default='status',choices=['language','genres','status','type'],help='Choose statistics by language, genres, status or type')

def one_day_update(x):
    if (datetime.datetime.now() - datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')).total_seconds()<=24*60*60:
        return 'yes'
    else:
        return 'no'


@api.route('/tv-shows/statistics')
class TVshowsStatistics(Resource):

    @api.response(404, 'TV shows were not found')
    @api.response(400, 'Validation Error')
    @api.response(200, 'OK')
    @api.doc(description="Retrieve statistics from TV shows. Given the 'by' parameter it can be seen the percentage \
             of a determine feature considering the total number of TV shows in the database")
    @api.expect(parser_3)
    def get(self):
        
        args = parser_3.parse_args()
        # retrieve the query parameters
        format_param = args.get('format')
        by_param= args.get('by')
  
        #get the db information in a dataframe
        df_aux=read_from_sqlite(datab_name, tab_name)
        
        #Check if the db table was empty
        if df_aux.empty:
            return {"message":"There are no tv shows to retrieve"},404
        
        #Total number of tv shows
        total_n_shows=df_aux.shape[0]
        
        #Total number of updated tv shows in the last 24hrs
        df_aux['24hrs']=df_aux['last_update'].apply(one_day_update)
        
        total_one_day_updates= int(df_aux[df_aux['24hrs']=='yes']['24hrs'].count())
        
        #get parametres statistics
        if format_param=='json': 
            if by_param in ['language','status','type']:
                df_aux=df_aux.groupby([by_param]).count()
                df_aux['percentage']=df_aux['id'].apply(lambda x: x/total_n_shows).round(2)
                values_dict=json.loads(df_aux['percentage'].to_json(orient='index'))    
            else:
                #Create a column with lists of genres instead of json string
                df_aux['genres_list']=df_aux['genres'].apply(lambda x: json.loads(x)['str_json'])
                df_aux=df_aux[['id','genres_list']].explode('genres_list')
                df_aux=df_aux.groupby(['genres_list']).count()
                df_aux['percentage']=df_aux.apply(lambda x: x/total_n_shows).round(2)
                df_aux=df_aux.sort_values(by='percentage',ascending=False)
                values_dict=json.loads(df_aux['percentage'].to_json(orient='index'))
            
            dict_to_return={"total":total_n_shows,"total-updated":total_one_day_updates,"values":values_dict}

            return dict_to_return,200
        else:
            if by_param in ['language','status','type']:
                df_aux=df_aux.groupby([by_param]).count().copy()
                df_aux['percentage']=df_aux['id'].apply(lambda x: x/total_n_shows).round(3)
                df_aux=df_aux.sort_values(by='percentage',ascending=True)
                
                plt.rcdefaults()
                fig, ax = plt.subplots(figsize=(16,6))
                
                y_axis = df_aux.index
                y_pos = np.arange(len(y_axis))
                percentage = df_aux['percentage'].apply(lambda x: x*100)
                
                ax.barh(y_pos, percentage)
                ax.set_xlim(0,df_aux['percentage'].max()*100+10)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(y_axis)
                
                ax.set_xlabel('Percentage with respect to the total TV shows (%)',fontsize=15)
                ax.set_ylabel(by_param,fontsize=15)
                ax.set_title('Frequency of {} associated to the TV shows'.format(by_param),fontsize=18)

                for i in ax.patches:
                    ax.text(i.get_width()+0.05, i.get_y()+0.15,str(round(i.get_width(), 3))+'%', fontsize=12,color='dimgrey')
                
                plt.savefig("z5222810-Q6.png")
                
            else:
               #Create a column with lists of genres instead of json string
                df_aux['genres_list']=df_aux['genres'].apply(lambda x: json.loads(x)['str_json'])
                df_aux=df_aux[['id','genres_list']].explode('genres_list')
                df_aux=df_aux.groupby(['genres_list']).count()
                df_aux['percentage']=df_aux.apply(lambda x: x/total_n_shows).round(3)
                df_aux=df_aux.sort_values(by='percentage',ascending=True)
                
                plt.rcdefaults()
                fig, ax = plt.subplots(figsize=(15,8))
                
                genre = df_aux.index
                y_pos = np.arange(len(genre))
                percentage = df_aux['percentage'].apply(lambda x: x*100)
                
                ax.barh(y_pos, percentage)
                ax.set_xlim(0,df_aux['percentage'].max()*100+10)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(genre)
                
                ax.set_xlabel('Percentage with respect to the total TV shows (%)',fontsize=15)
                ax.set_ylabel('Genres',fontsize=15)
                ax.set_title('Frequency of genres associated to the TV shows',fontsize=18)
                ax.text(0.0, -0.13, '*A TV show could be associated with more than one genre.',
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes,
                        color='blue', fontsize=10)
                
                for i in ax.patches:
                    ax.text(i.get_width()+0.05, i.get_y()+0.15,str(round(i.get_width(), 3))+'%', fontsize=12,color='dimgrey')
            
                plt.savefig("z5222810-Q6.png")
        
            return send_file("z5222810-Q6.png",mimetype='image/png',cache_timeout=0)



if __name__ == '__main__':
    
    datab_name='z5222810.db'
    tab_name='TV_Shows'
    port_number=5000
    host_name='localhost'
    
    db_creation(datab_name)
    table_structure="""CREATE TABLE IF NOT EXISTS TV_Shows(id integer PRIMARY KEY,
    last_update text, tvmaze_id integer NOT NULL, name text, type text, language text,
    genres text, status text, runtime integer, premiered text, officialSite text,
    schedule text, rating text, weight integer, network text, summary text)"""
        
    create_table(datab_name, table_structure)
        
    # run the application
    app.run(debug=True, port=port_number, host=host_name)
    
