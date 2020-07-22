from bokeh.plotting import figure, output_file, show

x=[1,2,3]
y=[4,2,1]
output_file('index.html')

p = figure(
    title='coba', x_axis_label='X Axis', y_axis_label='Y Axis'
)
p.line(x, y, legend='Test', line_width=2)
show(p)