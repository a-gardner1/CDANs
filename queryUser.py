# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 11:09:29 2016

@author: Drew
"""

import sys

def queryUser(question):
    """
    Ask a simple yes/no question of the user.
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    while True:
        try:
            sys.stdout.write(question + '[y/n]')
            choice = raw_input().lower()
            if choice == '':
                return valid['y']
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")
        except KeyboardInterrupt:
            # turns out this doesn't fix the problem with IPython console
            # console freezes if Ctrl-C during raw-input()
            sys.stdout.write("'No' answer assumed.")
            return False