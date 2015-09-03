import matplotlib.pyplot as plt
import logbook

log = logbook.Logger('Algorithm')


def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('price in $')

    # If data has been record()ed, then plot it.
    # Otherwise, log the fact that no data has been recorded.
    if 'AAPL' in perf and 'short_ema' in perf and 'long_ema' in perf:
        perf[['AAPL', 'short_ema', 'long_ema']].plot(ax=ax2)

        ax2.plot(perf.ix[perf.buy].index, perf.short_ema[perf.buy],
                 '^', markersize=10, color='m')
        ax2.plot(perf.ix[perf.sell].index, perf.short_ema[perf.sell],
                 'v', markersize=10, color='k')
        plt.legend(loc=0)
        plt.gcf().set_size_inches(18, 8)
    else:
        msg = 'AAPL, short_ema and long_ema data not captured using record().'
        ax2.annotate(msg, xy=(0.1, 0.5))
        log.info(msg)

    plt.show()
