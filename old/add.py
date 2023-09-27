import xml.etree.ElementTree as ET
import os
ad = 100
base = 39600

tree = ET.parse(os.path.join('../sumo_sim', 'test.rou.xml'))
root = tree.getroot()

# create the new vehicle element
vehicle = ET.Element("vehicle", id="ego", type="standard_car", depart=str(ad+base), departLane="free", departSpeed="max")
route = ET.Element("route", edges="327975022 39704919#0 542650138 542650138.26 266507547#0 -266507550#0 -308849647#1 -308849647#0 -201789794#3 -201789794#2 -201789794#1 -202070429#13 -202070429#12 -202070429#12.49 -202070429#11 -202070429#8 -202070429#5 -202070429#4 -202070429#2 -202070429#1 -202070429#0 151167728 151167728.19 257907860 814691984#0 814691984#1 305390325#0 305390325#2 305390325#3 266567304#0 64464420 25955089 64464430 814691990 814691990.7 -24991571 -24991571.15 -24991573 -11014139#1 145069303#1")
vehicle.append(route)

# find the "routes" element and append the new vehicle element to it
for i,v in enumerate(root):
    if 'depart' in v.keys():
        if float(v.attrib['depart']) > (base + ad):
            print(i, v.attrib['depart'])
            break

root.insert(i,vehicle)
# write the modified XML to a new file
tree.write("modified_file.xml")