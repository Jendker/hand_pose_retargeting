import rospy
from std_msgs.msg import String

pub = rospy.Publisher('test_topic', String, queue_size=10)
rospy.init_node('publisher')
r = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    pub.publish("hello world")
    r.sleep()