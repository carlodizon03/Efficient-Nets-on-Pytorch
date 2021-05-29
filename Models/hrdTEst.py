def get_link(layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
       
       
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i           # 2^n
          if layer % dv == 0:   # if 2^2 divides layer k:
            k = layer - dv      # then new layer to connect = k - 2^n
            link.append(k)      # connect layer k to the layer k - 2^n
            if i > 0:
                out_channels *= grmul
            print(dv, k, out_channels)
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        print("-----")


        for i in link:
          ch,_,_ = get_link(i, base_ch, growth_rate, grmul) #recursion
          in_channels += ch
        #   print(in_channels, ch)
        return out_channels, in_channels, link


print(get_link(4,24,16,1.6))



# layer = 2
# outch = 16
# gr = 1.6
# for i in range(10):
#     dv = 2**i
#     if layer % dv == 0 :
#         print(layer-dv)
# #         if i>0:
# #             outch*=gr
# #             print( outch)
# # outch = int(int(outch + 1)/2)*2
# # print(outch)