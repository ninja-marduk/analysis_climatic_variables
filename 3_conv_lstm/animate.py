def animate(i):
    array = da[i,:,:].values
    im.set_array(array.flatten())
    ax.set_title(f'Organic Matter AOD at 550nm, {str(da.time[i].values)[:-16]}', fontsize=12)