import xml.etree.cElementTree as ET

x0 = 0
x1 = 1
elements = 50
dx = (x1 - x0) / elements

nodes = elements + 1

nektar = ET.Element("NEKTAR")
geometry = ET.SubElement(nektar, "GEOMETRY", DIM="1" ,SPACE="1")
vertex = ET.SubElement(geometry, "VERTEX")

for i in range(nodes):
    vertex_id = ET.SubElement(vertex, "V", ID=f"{i}").text=f"{i*dx:4.4f} {0:4.4f} {0:4.4f}"

element = ET.SubElement(geometry, "ELEMENT")
for i in range(elements):
    element_id = ET.SubElement(element, "S", ID=f"{i}").text=f"{i:8d} {(i+1):8d}"

composite = ET.SubElement(geometry, "COMPOSITE")

# Fluid domain
composite_id = ET.SubElement(composite, "C", ID=f"0").text=f"S[0-{elements-1}]"
# Left BC
composite_id = ET.SubElement(composite, "C", ID=f"1").text=f"V[0]"
# Right BC
composite_id = ET.SubElement(composite, "C", ID=f"2").text=f"V[{nodes -1}]"

# Domain ID
domain = ET.SubElement(geometry, "DOMAIN")

domain_id = ET.SubElement(domain, "D", ID=f"0").text=f"C[0]"

tree = ET.ElementTree(nektar)
ET.indent(tree, space="\t", level=0)
tree.write("Leblanc1D.xml")
