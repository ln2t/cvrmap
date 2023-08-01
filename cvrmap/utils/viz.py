"""
Visualization utilities
"""

# imports

# classes

# functions

def gather_figures(objs):
    """
    Create fig from several DataObj of 'timecourse' type by superimposing their individual plots
    If two objects are given, the first one will have its y-axis on the left and the second on the right.
    """

    if len(objs) == 2:
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        counter = 0
        for obj in objs:
            if counter == 0:
                secondary_y = True
            else:
                secondary_y = False
            fig.add_trace(obj.figs['plot'].data[0], secondary_y=secondary_y)
            plot_title = obj.figs['plot'].layout.title.text
            plot_units = obj.figs['plot'].layout.yaxis.title.text
            title_text = plot_title + ', ' + plot_units
            fig.update_yaxes(title_text=title_text, secondary_y=secondary_y)
            counter = counter + 1
    else:
        import plotly.graph_objects as go
        fig = go.Figure()
        for obj in objs:
            fig.add_trace(obj.figs['plot'].data[0])
    return fig