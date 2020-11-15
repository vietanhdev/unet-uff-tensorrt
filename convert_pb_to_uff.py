import uff

frozen_filename ='./lane_segmentation_384x384.pb'
output_node_names = ['sigmoid/Sigmoid']
output_uff_filename = './lane_segmentation_384x384.uff'

uff_mode = uff.from_tensorflow_frozen_model(frozen_filename, output_nodes=output_node_names, output_filename=output_uff_filename, text=False)
