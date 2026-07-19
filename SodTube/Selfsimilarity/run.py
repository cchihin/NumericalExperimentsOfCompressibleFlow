import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from SodSolution import sod_exact

def generate_csv(path, session, chk, i, factor):
    
    xend = 1*factor
    xstart = -1*factor
    fieldconvert = f"FieldConvert -f -m interppoints:fromxml={session}:fromfld={chk}:line=201,{xstart},0,{xend},0 points_{i}.csv"
    subprocess.run(fieldconvert.split(), cwd=path)

def add_geometry(root, elements, x0=0.0, x1=1.0):

    nodes = elements + 1
    dx = (x1 - x0) / elements

    geometry = root.find("GEOMETRY")  # already created above

    # VERTEX
    vertex = ET.SubElement(geometry, "VERTEX")
    for i in range(nodes):
        x = x0 + i * dx
        ET.SubElement(vertex, "V", ID=str(i)).text = f"{x:.6f} 0.000000 0.000000"

    # ELEMENT
    element = ET.SubElement(geometry, "ELEMENT")
    for i in range(elements):
        ET.SubElement(element, "S", ID=str(i)).text = f"{i} {i+1}"

    # COMPOSITE
    composite = ET.SubElement(geometry, "COMPOSITE")

    # Fluid domain composite
    ET.SubElement(composite, "C", ID="0").text = f"S[0-{elements-1}]"

    # Left boundary
    ET.SubElement(composite, "C", ID="1").text = "V[0]"

    # Right boundary
    ET.SubElement(composite, "C", ID="2").text = f"V[{nodes-1}]"

    # Domain ID
    domain = ET.SubElement(geometry, "DOMAIN")
    domain_id = ET.SubElement(domain, "D", ID=f"0").text=f"C[0]"


    # -----------------------------------------------------
    # Call the function to add elements
    # -----------------------------------------------------

Nel = 16 

P = 12

run = 1

mu0 = 0.1

factorlist = [1, 2]

for factor in factorlist:

    # Creating folders
    # if os.path.exists(f"factor{factor}"):
    #     print(f"factor{factor}/ exists")
    # else:
    #     os.mkdir(f"factor{factor}")

    # # Reading session file
    # tree = ET.parse('SodTubeSession.xml')
    # root = tree.getroot()

    # # Modifying P
    # for exp in root.findall(".//EXPANSIONS/E"):
    #     if "NUMMODES" in exp.attrib:
    #         exp.set("NUMMODES", f"{P+1}")

    # # Adding Mesh
    # geometry = ET.SubElement(root, "GEOMETRY", DIM="1", SPACE="1")
    # add_geometry(root, elements=Nel, x0=-1.0*factor, x1=1.0*factor)

    # ET.indent(tree, space="\t", level=0)

    # # Saving .xml
    # tree.write(f"factor{factor}/SodTube_P{P}_factor{factor}.xml", encoding="utf-8", xml_declaration=True)

    # # Running simulation
    # cmprsblflw = f"CompressibleFlowSolver SodTube_P{P}_factor{factor}.xml"
    # if run:
    #     subprocess.run(cmprsblflw.split(), cwd=f"factor{factor}")
    # else:
    #     print(cmprsblflw)

    # Processing 
    for t in range(5):
        generate_csv(f"factor{factor}", f"SodTube_P{P}_factor{factor}.xml", f"SodTube_P{P}_factor{factor}_{t}.chk", t, factor)


# Plotting
for t in range(5):
    fig, ax = plt.subplots(2,3)
    
    for factor in factorlist:

        x = np.linspace(-1*factor, 1*factor, 400)

        rho, u_vel, p, E = sod_exact(x, factor*0.1*t)

        data = np.genfromtxt(f'factor{factor}/points_{t}.csv', delimiter=',', skip_header=1)

        ax[0,0].plot(x/factor, rho, 'k--')
        ax[0,0].plot(data[:,0]/factor, data[:,2], label=f"factor={factor}")
        ax[0,0].set_xlabel(r'$x/factor$')
        ax[0,0].set_ylabel(r'$\rho$')

        ax[0,1].plot(x/factor, u_vel, 'k--')
        ax[0,1].plot(data[:,0]/factor, data[:,5])
        ax[0,1].set_xlabel(r'$x/factor$')
        ax[0,1].set_ylabel(r'$u$')
        ax[0,1].grid()

        ax[1,0].plot(x/factor, p, 'k--')
        ax[1,0].plot(data[:,0]/factor, data[:,6])
        ax[1,0].set_xlabel(r'$x/factor$')
        ax[1,0].set_ylabel(r'$P$')
        ax[1,0].grid()

        ax[1,1].plot(x/factor, E, 'k--')
        ax[1,1].plot(data[:,0]/factor, data[:,4])
        ax[1,1].set_xlabel(r'$x/factor$')
        ax[1,1].set_ylabel(r'$E$')
        ax[1,1].grid()

        ax[0,2].plot(data[:,0]/factor, data[:,-1])
        ax[0,2].set_xlabel(r'$x/factor$')
        ax[0,2].set_ylabel(r'$Art. Vis.$')
        ax[0,2].grid()

        ax[1,2].plot(data[:,0]/factor, data[:,-2])
        ax[1,2].set_xlabel(r'$x/factor$')
        ax[1,2].set_ylabel(r'$Sensor$')
        ax[1,2].grid()

    ax[0,0].grid()
    ax[1,0].grid()
    ax[0,1].grid()
    ax[1,1].grid()
    ax[0,2].grid()
    ax[1,2].grid()

    ax[0,0].legend(ncol=3, loc="lower left", bbox_to_anchor=(1.2,1))
    figname = f'plot_{t}.png'
    fig.suptitle(fr'$N_{{el}} = {Nel}, t = {t*0.1:.1f}factor, \mu_0 = {mu0}$')
    print(f'Saving {figname}')
    fig.set_size_inches(12,6)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig(f'plot_{t}.png', dpi=200, bbox_inches='tight')
    plt.close('all')
    print(f'Done')

        # Plotting
