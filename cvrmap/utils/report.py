"""
Define Report Class to handle html reports
"""

# imports

# classes


class Report:
    """
    A class to create, update, modify and save reports with pretty stuff
    """
    def __init__(self, path=None, string=""):
        """
        path is the place where the report is written/updated
        """
        self.path = path
        self.string = string

    def init(self, subject, date_and_time, version,
             cmd):
        """
        Init the report with html headers and various information on the report
        """
        intro_part1 = """<!DOCTYPE html>
            <html>
            <body>
            <h1>CVRmap: individual report</h1>
            <h2>Summary</h2>
            <div>
            <ul>
            <li>BIDS participant label: sub-001</li>
            <li>Date and time: 2022-06-01 13:54:41.301278</li>
            <li>CVRmap version: dev
            </li>
            <li>Command line options:
            <pre>
            <code>
            """
        intro_part2 = """</code>
            </pre>
            </li>
            </ul>
            </div>
        """
        self.string.join(intro_part1 + str(cmd) + '\n' + intro_part2)
        with open(self.path, "w") as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<body>\n')
            f.write('<h1>CVRmap: individual report</h1>\n')
            f.write('<h2>Summary</h2>\n')
            f.write('<div>\n')
            f.write('<ul>\n')
            f.write(
                '<li>BIDS participant label: sub-%s</li>\n' % subject)
            f.write('<li>Date and time: %s</li>\n' % date_and_time)
            f.write('<li>CVRmap version: %s</li>\n' % version)
            f.write('<li>Command line options:\n')
            f.write('<pre>\n')
            f.write('<code>\n')
            f.write(str(cmd) + '\n')
            f.write('</code>\n')
            f.write('</pre>\n')
            f.write('</li>\n')
            f.write('</ul>\n')
            f.write('</div>\n')

    def add_section(self, title):
        """
        Add a section to the report
        """
        with open(self.path, "a") as f:
            f.write('<h2>%s</h2>\n' % title)

    def add_subsection(self, title):
        """
        Add a subsection to the report
        """
        with open(self.path, "a") as f:
            f.write('<h3>%s</h3>\n' % title)

    def add_sentence(self, sentence):
        """
        Add a sentence to the report
        """
        with open(self.path, "a") as f:
            f.write('%s<br>\n' % sentence)

    def add_image(self, fig):
        """
        Add an image path to the report

        The image is embedded in the hmtl report using base64
        fig is a Figure object from plotly
        """
        import base64
        # with open(self.path, "ab") as f:
        #   f.write(b'<img src = "data:image/png;base64, ')
        #   f.write(base64.b64encode(fig.to_image('png')))
        #   f.write(fig.to_html(full_html=False))
        #   f.write(b'" >')
        #   f.write(b'<br>')

        with open(self.path, "a") as f:
            f.write(fig.to_html(full_html=False))
            f.write('<br>')


    def finish(self):
        """
        Writes the last lines of the report to finish it.
        """
        with open(self.path, "a") as f:
            f.write('</body>\n')
            f.write('</html>\n')