
import sys, os

class Process(object):
    
    def __init__(self, variable_dict=None):
        super(Process, self).__init__()
        if variable_dict:
            self._instantiate(variable_dict)
        
            
    def _instantiate(self, variable_dict):
        for key, value in variable_dict.items():
            setattr(self, key, value)
            
    
    def _apply_defaults(self, defaults):
        for key, value in defaults.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)
            elif key in variable_dict:
                if variable_dict[key] is None:
                    setattr(self, key, value)
            
    
    def _assign_variables(self, variable_dict):
        for key, value in variable_dict.items():
            if value is not None:
                setattr(self, key, value)
            
    
    def _check_variables(self, variable_dict):
        proceed = True
        for key in variable_dict.keys():
            if getattr(self, key, None) is None:
                print 'ERROR: %s is set to None.' % key
                proceed = False
        return proceed


