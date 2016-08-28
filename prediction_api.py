# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:02:45 2016

@author: rahulkumar
"""

import web
import run_model



urls = (
    '/supply', 'index'
)

class index:
    def GET(self):
        web.header('Content-Type','application/json') 
        
        requirements = web.input(value=' ')

        try:
            output = run_model.model(requirement = [requirements.value])
        except:
            output = 'Invalid query'
            pass
        
        return output

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
