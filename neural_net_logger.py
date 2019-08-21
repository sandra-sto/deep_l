import tensorflow as tf


class NeuralNetLogger:

    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.writer = self.create_writer()

    def create_writer(self, log_dir):
        # log_dir = '%s_%i_%i_%s_%s' % (paths.get_log_dir(),
        #                               self.neural_net.hidden_nodes[0], self.neural_net.hidden_nodes[1],
        #                               self.neural_net.activation_functions[0],self.neural_net.activation_functions[1])
        writer = tf.summary.FileWriter(log_dir)

        writer.add_graph(self.neural_net.session.graph)
        return writer

    def log_performance(self, states, target_q, iteration):
        if iteration % 2 == 0:
            merged = tf.summary.merge_all()

            summary = self.neural_net.session.run(merged, feed_dict={self.neural_net.states: states, self.neural_net.target_q: target_q})
            self.writer.add_summary(summary, iteration)



    def summarize(self, step, summarizer, scope="", summaries=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.summary_dict[summarizer]
        if summaries is not None:
            summary_writer.add_summary(summaries, step)
        summary_writer.flush()
