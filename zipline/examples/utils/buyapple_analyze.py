import matplotlib.pyplot as plt
import logbook

log = logbook.Logger('Algorithm')


def analyze(context, perf):
    ax1 = plt.subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')
    ax2 = plt.subplot(212, sharex=ax1)
    # Check whether AAPL data has been record()ed.
    # If it has been, then plot it. Otherwise log
    # fact that it has not been recorded.
    if 'AAPL' in perf.columns:
        perf.AAPL.plot(ax=ax2)
        ax2.set_ylabel('AAPL price in $')
    else:
        ax2.annotate('AAPL data not captured using record().', xy=(0.35, 0.5))
        log.info('AAPL data not captured using record().')
    plt.gcf().set_size_inches(18, 8)
    plt.show()
