# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:02:45 2016

@author: rahulkumar
"""

import ijson
import pandas as pd
import numpy as np

'''
Read json, 
select variables, 
pre-process data
'''
csvdata = pd.read_json('dataset.json')
csvdata = csvdata.drop(['details','pieces', 'referenceno', 'tempwaybillno','waybillno' ],1)
csvdata['producttype'] = csvdata['producttype'].map({'T1':1.0,'T2':2.0,'T3':3.0,'T4':4.0,'T148':148.0,'T172':172.0,'T196':196.0,'T21400':21400.0,'T21100':21100.0})

writer = pd.ExcelWriter('Raw_data.xlsx', engine ='xlsxwriter')





f =  open('dataset.json', 'r')

object = ijson.items(f,'details')

index = 1000000
'''
Extract Delivery time
'''
data =[]
for item in object:
    for row in xrange(0,700000):
        try:
            data.append([[str(index+row)],item[str(index+row)][0][0], item[str(index+row)][0][2], item[str(index+row)][-1][0], item[str(index+row)][-1][2]])
        except:
            data.append([[str(index+row)],[0.0],[0.0],[0.0],[0.0]])
            pass
    

from dateutil import parser

'''
Extract total hours betweentaken to deliver product
'''
overall_hours= []
for i in xrange (0,len (data)):
    try:
        #convert string to actual date and time
        dt1 = parser.parse(data[i][1]+' ' + data[i][2])
        dt2 = parser.parse(data[i][3]+' ' + data[i][4])
        
        diff = dt1 - dt2
        
        #days
        days = diff.days
         
        #overall hours
        days_to_hours = days * 24
        diff_btw_two_times = (diff.seconds) / 3600
        overall_hours.append( days_to_hours + diff_btw_two_times)
    except:
        overall_hours.append(0.0)
        pass

csvdata['overall_hours'] = np.array(overall_hours)
csvdata.to_excel(writer, sheet_name = 'Sheet1')

writer.save()
f.close()