import os
import numpy as np
import os
import io
import traci
import time
import numpy as np
from xml.dom import minidom
from collections import deque
import sumolib
from sumolib import checkBinary
from lxml.etree import XMLParser
import xml.etree.ElementTree as ET
import json
import random
import pandas as pd


def generate_route_file(rou_file):
    rou_dict = json.load(open(rou_file))
    for rou in rou_dict:
        diff = random.randint(-5, +5)
        rou_dict[rou]['arrival_time'] += diff
    rou_df = pd.DataFrame(rou_dict).T
    rou_df = rou_df.sort_values(by=['arrival_time'])
    rou_dict = rou_df.T.to_dict()
    with open(os.path.join('SUMO_gym','sumo_ingolstadt',"adaptive_route.rou.xml"), "w") as routes:
        print("""<routes>
        <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" /> """, file=routes)

        for rou in rou_dict:
            diff = random.randint(-5, +5)
            edge_str = ''
            for e in rou_dict[rou]['edges']:
                edge_str = edge_str + str(e) + ' '
            print(f"""
            <vehicle
            id = "{rou}"
            type = "standard_car"
            depart = "{rou_dict[rou]['arrival_time']}"
            departLane = "free"
            departSpeed = "max">
                          <route
            edges = "{edge_str}"/>
                    </vehicle >
            """, file=routes)
        print("</routes>", file=routes)

if __name__ == '__main__':
    generate_route_file('rou_dict.json')