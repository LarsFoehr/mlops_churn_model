"""Unit tests - Example
    
    * This script contains one example unit test. Further unit tests wlll be added.
    * form_request from the flask app is tested, if it gives the right response in case of incorrect input data.
    
    * You can run the test(s) via: pytest -v
"""
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from app import form_response, NotANumber 

# Build the input data, which will be used inside the unit test
# The data is a dictionary, which contains one example of incorrect and one example of correct data.
input_data = {
    "incorrect_values": 
    
    {"number_vmail_messages": 3, 
    "total_day_calls": 4, 
    "total_eve_minutes": 'as', 
    "total_eve_charge": 12, 
    "total_intl_minutes": 1, 
    "number_customer_service_calls": 'ab', 
    },

    "correct_values": 
    {"number_vmail_messages": 3, 
    "total_day_calls": 4, 
    "total_eve_minutes": 2, 
    "total_eve_charge": 12, 
    "total_intl_minutes": 1, 
    "number_customer_service_calls": 4, 
    }
}

def test_form_response_incorrect_values(data = input_data["incorrect_values"]):
    res = form_response(data)
    assert res == NotANumber().message