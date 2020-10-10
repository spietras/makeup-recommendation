def first_channel_ordering(labels, pixels):
    mean_ls = {label: pixels[labels == label].mean(axis=0)[0] for label in range(max(labels) + 1)}
    return sorted(mean_ls, key=mean_ls.get)