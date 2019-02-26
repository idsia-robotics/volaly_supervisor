#!/usr/bin/env python

import copy
import math
import threading
import numpy as np

import rospy
import smach
import smach_ros

import tf2_ros
import tf2_geometry_msgs

import PyKDL as kdl
import message_filters as mf
import tf_conversions as tfc

from std_msgs.msg import Bool, ColorRGBA, Duration, Float32, String
from geometry_msgs.msg import PoseStamped

from neopixel_ble.msg import NeoPixelColor

class FsmNode():
    def __init__(self):
        self.beacons_set = set(rospy.get_param('~beacons_set', [1, 2, 3]))
        self.beacons_ns = rospy.get_param('~beacons_ns', '/')
        self.beacons_prefix = rospy.get_param('~beacons_prefix', 'beacon')

        # Next target
        c_next = rospy.get_param('~color_next', {'index': 255, 'color': {'r': 0.0, 'g': 1.0, 'b': 1.0, 'a': 0.1}})
        self.color_next = NeoPixelColor(index=c_next['index'], color=ColorRGBA(**c_next['color']))

        # Wait for confirmation
        c_wait = rospy.get_param('~color_wait', {'index': 255, 'color': {'r': 1.0, 'g': 0.4, 'b': 0.0, 'a': 0.1}})
        self.color_wait = NeoPixelColor(index=c_wait['index'], color=ColorRGBA(**c_wait['color']))

        # Confirmed
        c_confirm = rospy.get_param('~color_confirm', {'index': 255, 'color': {'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 0.2}})
        self.color_confirm = NeoPixelColor(index=c_confirm['index'], color=ColorRGBA(**c_confirm['color']))

        # Complete
        c_complete = rospy.get_param('~color_complete', {'index': 255, 'color': {'r': 0.0, 'g': 1.0, 'b': 1.0, 'a': 0.2}})
        self.color_complete = NeoPixelColor(index=c_complete['index'], color=ColorRGBA(**c_complete['color']))

        # Clear pixels
        self.color_clear = NeoPixelColor(index=255, color=ColorRGBA())

        relloc_fsm_state_topic = rospy.get_param('~relloc_fsm_state_topic', 'drone_relloc_fsm/state')
        self.sub_relloc_fsm_state = rospy.Subscriber(relloc_fsm_state_topic, String, self.relloc_fsm_state_cb)
        self.relloc_fsm_event = threading.Event()
        self.relloc_fsm_state = None

        self.tf_buff = tf2_ros.Buffer()
        self.tf_ls = tf2_ros.TransformListener(self.tf_buff)

        robot_current_pose_topic = rospy.get_param('~robot_current_pose_topic', '/optitrack/robot')
        self.sub_current_pose = rospy.Subscriber(robot_current_pose_topic, PoseStamped, self.robot_current_pose_cb)
        self.robot_current_pose = None

        self.ignore_z = rospy.get_param('~ignore_z', True)
        self.target_threshold_xy = rospy.get_param('~target_threshold_xy', 0.10) # 10cm
        self.target_threshold_z = rospy.get_param('~target_threshold_z', 0.20) # 10cm

        self.state_name_topic = rospy.get_param('~state_name_topic', '~state')
        self.pub_state_name = rospy.Publisher(self.state_name_topic, String, queue_size = 10, latch = True)

        self.pubs_beacon_color = {it: rospy.Publisher(self.beacons_ns + self.beacons_prefix + str(it) + '/color',
                                            NeoPixelColor, queue_size = 10) for it in self.beacons_set}

        self.sm = smach.StateMachine(outcomes = ['FINISH'])

        initial_tt = np.random.choice(list(self.beacons_set), size = len(self.beacons_set), replace = False)
        self.sm.userdata.targets_list = initial_tt.tolist()
        rospy.loginfo('Targets list: {}'.format(self.sm.userdata.targets_list))

        with self.sm:
            # smach.StateMachine.add('INIT_ALL_CLEAR',
            #     smach.CBState(self.set_all_color, cb_kwargs = {'context': self, 'color': self.color_complete, 'delay': 1.0}),
            #     transitions = {'done': 'WAIT_FOR_RELLOC',
            #                    'preempted': 'FINISH'})

            smach.StateMachine.add('WAIT_FOR_RELLOC',
                smach.CBState(self.wait_for_relloc_fsm_state, cb_kwargs = {'context': self, 'expected_state': 'FOLLOW_POINTING'}),
                transitions = {'ready': 'CHOOSE_NEXT',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('CHOOSE_NEXT',
                smach.CBState(self.choose_next_target, cb_kwargs = {'context': self}),
                transitions = {'done': 'INDICATE_NEXT',
                               'empty': 'WAIT_FOR_LANDING',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('INDICATE_NEXT',
                smach.CBState(self.set_color, cb_kwargs = {'context': self, 'color': self.color_next}),
                transitions = {'done': 'WAIT_TO_ENTER',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('WAIT_TO_ENTER',
                smach.CBState(self.wait_to_enter, cb_kwargs = {'context': self}),
                transitions = {'entered': 'INDICATE_WAIT',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('INDICATE_WAIT',
                smach.CBState(self.set_color, cb_kwargs = {'context': self, 'color': self.color_wait}),
                transitions = {'done': 'WAIT_TO_CONFIRM',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('WAIT_TO_CONFIRM',
                smach.CBState(self.wait_to_confirm, cb_kwargs = {'context': self}),
                transitions = {'confirmed': 'INDICATE_CONFIRMED',
                               'canceled': 'INDICATE_NEXT',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('INDICATE_CONFIRMED',
                smach.CBState(self.set_color, cb_kwargs = {'context': self, 'color': self.color_confirm, 'delay': 0.5}),
                transitions = {'done': 'INDICATE_CLEAR',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('INDICATE_CLEAR',
                smach.CBState(self.set_color, cb_kwargs = {'context': self, 'color': self.color_clear}),
                transitions = {'done': 'CHOOSE_NEXT',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('WAIT_FOR_LANDING',
                smach.CBState(self.wait_for_relloc_fsm_state, cb_kwargs = {'context': self, 'expected_state': 'LAND'}),
                transitions = {'ready': 'INDICATE_ALL_COMPLETE',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('INDICATE_ALL_COMPLETE',
                smach.CBState(self.set_all_color, cb_kwargs = {'context': self, 'color': self.color_complete, 'delay': 3.0}),
                transitions = {'done': 'INDICATE_ALL_CLEAR',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('INDICATE_ALL_CLEAR',
                smach.CBState(self.set_all_color, cb_kwargs = {'context': self, 'color': self.color_clear}),
                transitions = {'done': 'WAIT_FOR_RELLOC',
                               'preempted': 'FINISH'})

            # smach.StateMachine.add('FINISH',
            #     smach.CBState(self.set_all_color, cb_kwargs = {'context': self, 'color': self.color_clear}),
            #     transitions = {'done': 'OVER',
            #                    'preempted': 'OVER'})

        self.sm.register_start_cb(self.state_transition_cb, cb_args = [self.sm])
        self.sm.register_transition_cb(self.state_transition_cb, cb_args = [self.sm])
        self.sis = smach_ros.IntrospectionServer('smach_server', self.sm, '/SM_BEACONS')

    def relloc_fsm_state_cb(self, msg):
        self.relloc_fsm_state = msg.data
        self.relloc_fsm_event.set()

    def state_transition_cb(self, *args, **kwargs):
        states = args[2].get_active_states()
        s = '-'.join(states)

        # rospy.loginfo(args[0].__dict__)

        self.pub_state_name.publish(String(s))

    def robot_current_pose_cb(self, msg):
        # E.g. /optitrack/bebop in 'World' frame
        self.robot_current_pose = msg

    def distance_to_target(self, target_id):
        target_frame_id = self.beacons_prefix + str(target_id)

        if not self.robot_current_pose:
            return float('inf'), float('inf')

        try:
            # Beacon in robot's frame
            t = self.tf_buff.lookup_transform(self.robot_current_pose.header.frame_id, target_frame_id, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
            rospy.logerr_throttle(1.0, e)
            return float('inf'), float('inf')

        robot_p = tfc.fromMsg(self.robot_current_pose.pose)
        beacon_p = tf2_geometry_msgs.transform_to_kdl(t)

        # Ignore z-axis?
        # if self.ignore_z:
        d_z = np.fabs(beacon_p.p[2] - robot_p.p[2])

        robot_p.p[2] = 0.0
        beacon_p.p[2] = 0.0

        d_xy = (beacon_p.p - robot_p.p).Norm()

        return d_xy, d_z

    @smach.cb_interface(output_keys = ['targets_list'], outcomes = ['ready', 'preempted'])
    def wait_for_relloc_fsm_state(udata, context, expected_state):
        while not rospy.is_shutdown():
            if context.relloc_fsm_state == expected_state:
                return 'ready'

            # Sleep at max 0.1s, if there is an event it will wake up earlier
            context.relloc_fsm_event.wait(0.1)
            # Force to wait another event
            context.relloc_fsm_event.clear()

        return 'preempted'

    @smach.cb_interface(input_keys = ['targets_list'], output_keys = ['targets_list', 'target'], outcomes = ['done', 'empty', 'preempted'])
    def choose_next_target(udata, context):
        if not udata.targets_list:
            tt = np.random.choice(list(context.beacons_set), size = len(context.beacons_set), replace = False)
            udata.targets_list = tt.tolist()
            rospy.loginfo('Targets list: {}'.format(udata.targets_list))
            udata.target = None
            return 'empty'

        udata.target = udata.targets_list.pop()
        return 'done'

    @smach.cb_interface(input_keys = ['targets_list', 'target'], output_keys = ['targets_list', 'target'], outcomes = ['done', 'preempted'])
    def set_color(udata, context, color, delay = None):
        if udata.target in context.pubs_beacon_color:
            context.pubs_beacon_color[udata.target].publish(color)
            if delay:
                rospy.sleep(delay)

        return 'done'

    @smach.cb_interface(outcomes = ['done', 'preempted'])
    def set_all_color(udata, context, color, delay = None):
        for b_id, pub in context.pubs_beacon_color.items():
            pub.publish(color)
            rospy.loginfo(b_id)

        if delay:
            rospy.sleep(delay)

        return 'done'

    @smach.cb_interface(input_keys = ['targets_list', 'target'], output_keys = ['targets_list', 'target'], outcomes = ['entered', 'preempted'])
    def wait_to_enter(udata, context):
        loop_rate = rospy.Rate(30.0) # 30Hz

        while not rospy.is_shutdown():
            (xy, z) = context.distance_to_target(udata.target)
            if  xy < context.target_threshold_xy and z < context.target_threshold_z:
                return 'entered'
            loop_rate.sleep()

        return 'preempted'

    @smach.cb_interface(input_keys = ['targets_list', 'target'], output_keys = ['targets_list', 'target'], outcomes = ['confirmed', 'canceled', 'preempted'])
    def wait_to_confirm(udata, context):
        loop_rate = rospy.Rate(30.0) # 30Hz

        goal_time = rospy.Time.now() + rospy.Duration(2.0) # stay there at least 2s

        while not rospy.is_shutdown():
            (xy, z) = context.distance_to_target(udata.target)
            if  xy < context.target_threshold_xy and z < context.target_threshold_z:
                if rospy.Time.now() > goal_time:
                    return 'confirmed'
            else:
                return 'canceled'
            loop_rate.sleep()

        return 'preempted'

    def run(self):
        rospy.on_shutdown(lambda: self.sm.request_preempt())

        self.sis.start()

        smach_thread = threading.Thread(target = self.sm.execute)
        smach_thread.start()

        rospy.spin()
        smach_thread.join()

        self.sis.stop()

if __name__ == '__main__':
    rospy.init_node('beacons_fsm_node')

    fsm = FsmNode()

    try:
        fsm.run()
    except rospy.ROSInterruptException:
        rospy.logdebug('Exiting')
        pass
