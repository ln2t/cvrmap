# IMPORTS

# MAIN

def init(target, subject, date_and_time, version, command_line_arguments_dict):
    with open(target, "w") as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<body>\n')
        f.write('<h1>CVRmap: individual report</h1>\n')
        f.write('<h2>Summary</h2>\n')
        f.write('<div>\n')
        f.write('<ul>\n')
        f.write('<li>BIDS participant label: sub-%s</li>\n' % subject)
        f.write('<li>Date and time: %s</li>\n' % date_and_time)
        f.write('<li>CVRmap version: %s</li>\n' % version)
        f.write('<li>Command line options:\n')
        f.write('<pre>\n')
        f.write('<code>\n')
        f.write(str(command_line_arguments_dict) + '\n')
        f.write('</code>\n')
        f.write('</pre>\n')
        f.write('</li>\n')
        f.write('</ul>\n')
        f.write('</div>\n')
        f.close()

def add_section(target, title):
    with open(target, "a") as f:
        f.write('<h2>%s</h2>\n' % title)
        f.close()

def add_subsection(target, title):
    with open(target, "a") as f:
        f.write('<h3>%s</h3>\n' % title)
        f.close()

def add_sentence(target, sentence):
    with open(target, "a") as f:
        f.write('%s<br>\n' % sentence)
        f.close()

def add_image(target, image_path, **options):

    #todo: embed images using base64 source.png > source.txt and then <img src="data:image/png;base64, BIG_TEXT_HERE">

    width_option = ""
    for key in options.keys():
        if key == 'width':
            width_option = "width=" + options[key]

    with open(target, "a") as f:
        f.write('<img src="%s" %s><br>\n' % (image_path, width_option))
        f.close()

def finish(target):
    with open(target, "a") as f:
        f.write('</body>\n')
        f.write('</html>\n')
        f.close()
