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
            if fig is a str, then it is interpreted as a path to the image to save.
            If a plotly figure, uses the .to_html argument to embded the image

        Args:
            fig: str or Figure object from plotly
        """
        import base64
        # with open(self.path, "ab") as f:
        #   f.write(b'<img src = "data:image/png;base64, ')
        #   f.write(base64.b64encode(fig.to_image('png')))
        #   f.write(fig.to_html(full_html=False))
        #   f.write(b'" >')
        #   f.write(b'<br>')

        with open(self.path, "a") as f:
            if type(fig) is str:
                import os.path as path
                # make the path relative to report location
                _report_dir = path.dirname(self.path)
                _fig = path.relpath(fig, _report_dir)
                f.write('<figure> <img src=' + _fig + ' style="width:100%"> </figure>')
            else:
                f.write(fig.to_html(full_html=False))
            f.write('<br>')



    def finish(self):
        """
            Writes the last lines of the report to finish it.
        """
        with open(self.path, "a") as f:
            f.write('</body>\n')
            f.write('</html>\n')


# functions


def build_report(subject_label, args, __version__, physio, probe, baseline, global_signal, global_signal_shift, aroma_noise_ic_list, corrected_noise, fwhm, t1w, outputs, response, results):
    from datetime import datetime
    from .viz import gather_figures
    # Report
    # init
    report = Report(
        outputs['report'])
    report.init(subject=subject_label,
                date_and_time=datetime.now(),
                version=__version__, cmd=args.__dict__)

    if args.vesselsignal:
        report.add_subsection(title='Vessel signal')
        report.add_sentence(
            sentence="Vessel mask on mean epi:")
        report.add_image(outputs['vesselmask_figure'])
        report.add_sentence(
            sentence="Corresponding BOLD timecourse:")
        report.add_image(outputs['vesselsignal_figure'])
    else:
        if args.globalsignal:
            report.add_subsection(title='Global signal')
            report.add_sentence(
                sentence="Global mask on mean epi:")
            report.add_image(outputs['globalmask_figure'])
            report.add_sentence(
                sentence="Corresponding BOLD timecourse:")
            report.add_image(outputs['globalsignal_figure'])
        else:
            # Physio data
            # create figures for report

            report.add_subsection(title='Physiological data')
            report.add_sentence(
                sentence="Physiological data, with reconstructed "
                         "upper envelope and baseline:")
            report.add_image(outputs['breathing_figure'])

            # global signal and etco2
            report.add_section(title='Global Signal and etCO2')
            report.add_sentence(sentence="The computed shift is %s seconds" % global_signal_shift)

            # report.add_image(global_signal.figs['plot'])
            report.add_image(outputs['boldmean_figure'])

    # info on denoising
    report.add_section(title='Info on denoising')
    if not args.use_aroma:
        report.add_sentence(sentence="The noise regressors as classified by AROMA are (total: %s): %s" % (
        len(aroma_noise_ic_list), ','.join(aroma_noise_ic_list)))
        report.add_sentence(
            sentence="Keeping only those that do not correlate too much with the probe regressor gives the following list (total: %s): %s" % (
            #len(corrected_noise), ','.join(corrected_noise)))
            len(corrected_noise), str(corrected_noise)))
        report.add_sentence(sentence="Data are smoothed with a FWHM of %s mm" % fwhm)
        #report.add_sentence(sentence="Highpass filter cut-off set to %s Hz" % hpsigma)
    else:
        report.add_sentence(sentence="Denoised data from the AROMA classification of noise regressors")

    report.add_section(title='Results')

    # Delay map
    report.add_subsection(title='Delay map (units: s)')
    report.add_image(outputs['delay_figure'])
    # results['delay'].make_fig(fig_type='lightbox', **{'background': t1w})
    #report.add_image(results['delay'].figs['lightbox'])


    # report.add_sentence(
    #     sentence="Delay map histogram")
    # todo: add some stats (mean and std) of delay map
    # results['delay'].make_fig(fig_type='histogram')
    # report.add_image(results['delay'].figs['histogram'])

    # CVR
    report.add_subsection(title="CVR map %s" % results['cvr'].units)
    report.add_image(outputs['cvr_figure'])
    #response.make_fig(fig_type='lightbox')
    #report.add_image(response.figs['lightbox'])
    # finish the report
    report.finish()