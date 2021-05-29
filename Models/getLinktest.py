def get_link(layer, base_ch, growth_rate, grmul):
    if layer == 0:
        return base_ch, 0, []
    out_channels = growth_rate
    link = []
    for i in range(10):
        dv = 2 ** i
        if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
    out_channels = int(int(out_channels + 1) / 2) * 2
    in_channels = 0
    for i in link:
        ch,_,_ = get_link(i, base_ch, growth_rate, grmul)
        in_channels += ch
    return out_channels, in_channels, link
links = []
layers_ = []
out_channels = 0
n_layers = 4
for i in range(n_layers):
   outch, inch, link = get_link(i+1, 48, 10, 1.7)
   links.append(link)
   layers_.append([inch, outch])
   if (i % 2 == 0) or (i == n_layers - 1):
        out_channels += outch

print(links)
print(layers_)
print(out_channels)