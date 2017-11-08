"""
A simple script to start tensorflow servers with different roles.
"""
import tensorflow as tf

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

#
tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node-0.dongxiaoyuhuanyu.michigan-bigdata-pg0.wisc.cloudlab.us:2222"
    ],
    "worker" : [
        "node-1.dongxiaoyuhuanyu.michigan-bigdata-pg0.wisc.cloudlab.us:2222",
        "node-2.dongxiaoyuhuanyu.michigan-bigdata-pg0.wisc.cloudlab.us:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "host_name0:2222"
    ],
    "worker" : [
        "host_name0:2222",
        "host_name1:2222",
        "host_name2:2222",
        "host_name3:2222"
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
server.join()
