from Summary_Generator.Tensorflow_Graph import order_planner_without_copynet

# just execute the get_computation_graph function here:
graph = order_planner_without_copynet.get_computation_graph (
	seed_value = 3,

	# vocabulary sizes
	field_vocab_size = 100, 
	content_label_vocab_size = 100,

	# Embeddings size:
	field_embedding_size = 100,
	content_label_embedding_size = 400, # This is a much bigger vocabulary compared to the field_name's vocabulary

	# LSTM hidden state sizes
	lstm_cell_state_size = 500,
	hidden_state_size = 500 # they are same (for now)
)
