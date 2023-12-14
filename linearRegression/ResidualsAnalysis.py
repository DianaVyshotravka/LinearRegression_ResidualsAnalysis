from .LinearRegression import *
import matplotlib.pyplot as plt
import seaborn as sns

def residuals_distribution(model: LinearRegression, y, x, **plot_kw):
    """

    Args:
        model: LinearRegression object
        y: Array like
        x: Array like
        **plot_kw: seaborn style keyword arguments

    Returns: plotly figure object

    """
    residuals = y - model.predict(x)

    plot = sns.kdeplot(x=residuals, fill=True, **plot_kw)
    plot.set(xlabel='Residual', ylabel='Distribution', title='Residuals distribution')
    return plot

def prediction_plot(model: LinearRegression, y, x, **plot_kw):
    """
    Builds plot that shows Y yo Y predicted
    Args:
        model: LinearRegression object
        y: ArrayLike
        x: ArrayLike
        **plot_kw: seaborn style keyword arguments

    Returns:
        plotly figure object
    """
    predicted = model.predict(x)

    plot = sns.lineplot(x=y, y=y, color='black', alpha=0.5, **plot_kw)
    plot.scatter(x=predicted, y=y)
    plot.set(xlabel='Residual', ylabel='Y', title='Y to Y predicted')
    return plot

def residuals_plot(model: LinearRegression, y, x, line=True, **plot_kw):
    """
    Builds plot that shows Y to Y residuals
    Args:
        model: LinearRegression object
        y: ArrayLike
        x: ArrayLike
        line: bool, if True builds y = y line on graph
        **plot_kw: seaborn style keyword arguments

    Returns:
        plotly figure object
    """
    residuals = y - model.predict(x)
    plot = sns.scatterplot(y=y, x=residuals, **plot_kw)
    plot.set(xlabel='Residual', ylabel='Y', title='Y to Y residuals')
    if line:
        plot.axvline(color = 'black', alpha = 0.5)
    return plot

def residual_analysis(model: LinearRegression, y, x, **plot_kw):
    """
    Builds grid with 3 plots that show residual analysis
    Args:
        model: LinearRegression object
        y: ArrayLike
        x: ArrayLike
        **plot_kw: seaborn style keyword arguments

    Returns:
        plotly figure object
    """
    fig, axs = plt.subplots(nrows=3)
    fig.tight_layout(pad=3)

    residuals_distribution(model, x=x, y =y, ax=axs[2], **plot_kw)
    prediction_plot(model, y, x, ax=axs[0], **plot_kw)
    residuals_plot(model, y, x, ax=axs[1], **plot_kw)

    return (fig, axs)
