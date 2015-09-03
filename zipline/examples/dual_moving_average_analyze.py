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
    if 'AAPL' in perf and 'short_mavg' in perf and 'long_mavg' in perf:
        perf['AAPL'].plot(ax=ax2)
        perf[['short_mavg', 'long_mavg']].plot(ax=ax2)

        perf_trans = perf.ix[[t != [] for t in perf.transactions]]
        buys = perf_trans.ix[[t[0]['amount'] > 0 for t in
                              perf_trans.transactions]]
        sells = perf_trans.ix[
            [t[0]['amount'] < 0 for t in perf_trans.transactions]]
        ax2.plot(buys.index, perf.short_mavg.ix[buys.index],
                 '^', markersize=10, color='m')
        ax2.plot(sells.index, perf.short_mavg.ix[sells.index],
                 'v', markersize=10, color='k')
        plt.legend(loc=0)
    else:
        ax2.annotate('No data captured using record().', xy=(0.35, 0.5))
        log.info('No data captured using record().')

    plt.show()
