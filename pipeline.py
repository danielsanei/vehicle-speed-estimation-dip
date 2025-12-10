from graphviz import Digraph

def create_pipeline_graphic():
    # Initialize the graph
    dot = Digraph('Adaptive_Pipeline', comment='Adaptive Image Enhancement Pipeline')
    dot.attr(rankdir='LR', size='12,8', ratio='fill', splines='ortho')
    
    # --- Define Node Styles ---
    # Blue: Input/Output
    dot.attr('node', shape='box', style='filled', fillcolor='#DAE8FC', color='#6C8EBF', fontname='Helvetica', fontsize='12')
    dot.node('A', 'Raw Video\nFrame')
    dot.node('H', 'Enhanced\nFrame')
    dot.node('I', 'YOLOv8 Detector\n(Vehicle Detections)')

    # Yellow: Analysis Logic
    dot.attr('node', shape='diamond', style='filled', fillcolor='#FFF2CC', color='#D6B656')
    dot.node('B', 'Scene Analysis\n(Calculate Metrics)\nFog Density & Brightness')

    # Green: Processing Modules
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='#D5E8D4', color='#82B366')
    dot.node('C', 'Dehazing\n(Dark Channel Prior,\nω=0.85)')
    dot.node('D', 'Passthrough\n(No Change)')
    
    # Purple/Orange Gradient (Simulated with Box): Advanced Hybrid Contrast
    # We represent this as a subgraph to show it's a composite strategy
    with dot.subgraph(name='cluster_AHC') as c:
        c.attr(style='filled', color='#E1D5E7', fillcolor='#E1D5E7', label='Advanced Hybrid Contrast\n(Low Light Strategy)')
        c.node('E', 'CLAHE\n(Clip=3.0)', style='filled', fillcolor='#FFE6CC', color='#D79B00')
        c.node('F', 'LAB-space\nAdaptive Gamma', style='filled', fillcolor='#FFE6CC', color='#D79B00')
        # Logic flow inside the advanced block
        c.edge('E', 'F', label='Refinement')

    # --- Define Edges (The Logic Tree) ---
    dot.attr('edge', fontname='Helvetica', fontsize='10', color='#555555')

    # 1. Input to Analysis
    dot.edge('A', 'B')

    # 2. Fog Branch
    dot.edge('B', 'C', label='Fog Index > 0.30')
    dot.edge('C', 'H')

    # 3. Pristine Branch
    dot.edge('B', 'D', label='Else (Pristine)')
    dot.edge('D', 'H')

    # 4. Low Light / Advanced Contrast Branch
    dot.edge('B', 'E', label='Brightness < 80')
    dot.edge('F', 'H')

    # 5. Output to Detector
    dot.edge('H', 'I')

    return dot

# Generate and view the graph
# Note: You need Graphviz installed on your system (apt-get install graphviz)
pipeline_graph = create_pipeline_graphic()
pipeline_graph.render('adaptive_pipeline_graphic', format='png', cleanup=True)

print("Graphic generated: adaptive_pipeline_graphic.png")