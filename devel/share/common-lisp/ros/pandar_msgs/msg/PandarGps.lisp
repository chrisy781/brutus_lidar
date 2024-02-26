; Auto-generated. Do not edit!


(cl:in-package pandar_msgs-msg)


;//! \htmlinclude PandarGps.msg.html

(cl:defclass <PandarGps> (roslisp-msg-protocol:ros-message)
  ((stamp
    :reader stamp
    :initarg :stamp
    :type cl:real
    :initform 0)
   (used
    :reader used
    :initarg :used
    :type cl:integer
    :initform 0)
   (year
    :reader year
    :initarg :year
    :type cl:fixnum
    :initform 0)
   (month
    :reader month
    :initarg :month
    :type cl:fixnum
    :initform 0)
   (day
    :reader day
    :initarg :day
    :type cl:fixnum
    :initform 0)
   (hour
    :reader hour
    :initarg :hour
    :type cl:fixnum
    :initform 0)
   (minute
    :reader minute
    :initarg :minute
    :type cl:fixnum
    :initform 0)
   (second
    :reader second
    :initarg :second
    :type cl:fixnum
    :initform 0)
   (fineTime
    :reader fineTime
    :initarg :fineTime
    :type cl:integer
    :initform 0)
   (flag
    :reader flag
    :initarg :flag
    :type cl:integer
    :initform 0))
)

(cl:defclass PandarGps (<PandarGps>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <PandarGps>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'PandarGps)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name pandar_msgs-msg:<PandarGps> is deprecated: use pandar_msgs-msg:PandarGps instead.")))

(cl:ensure-generic-function 'stamp-val :lambda-list '(m))
(cl:defmethod stamp-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:stamp-val is deprecated.  Use pandar_msgs-msg:stamp instead.")
  (stamp m))

(cl:ensure-generic-function 'used-val :lambda-list '(m))
(cl:defmethod used-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:used-val is deprecated.  Use pandar_msgs-msg:used instead.")
  (used m))

(cl:ensure-generic-function 'year-val :lambda-list '(m))
(cl:defmethod year-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:year-val is deprecated.  Use pandar_msgs-msg:year instead.")
  (year m))

(cl:ensure-generic-function 'month-val :lambda-list '(m))
(cl:defmethod month-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:month-val is deprecated.  Use pandar_msgs-msg:month instead.")
  (month m))

(cl:ensure-generic-function 'day-val :lambda-list '(m))
(cl:defmethod day-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:day-val is deprecated.  Use pandar_msgs-msg:day instead.")
  (day m))

(cl:ensure-generic-function 'hour-val :lambda-list '(m))
(cl:defmethod hour-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:hour-val is deprecated.  Use pandar_msgs-msg:hour instead.")
  (hour m))

(cl:ensure-generic-function 'minute-val :lambda-list '(m))
(cl:defmethod minute-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:minute-val is deprecated.  Use pandar_msgs-msg:minute instead.")
  (minute m))

(cl:ensure-generic-function 'second-val :lambda-list '(m))
(cl:defmethod second-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:second-val is deprecated.  Use pandar_msgs-msg:second instead.")
  (second m))

(cl:ensure-generic-function 'fineTime-val :lambda-list '(m))
(cl:defmethod fineTime-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:fineTime-val is deprecated.  Use pandar_msgs-msg:fineTime instead.")
  (fineTime m))

(cl:ensure-generic-function 'flag-val :lambda-list '(m))
(cl:defmethod flag-val ((m <PandarGps>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader pandar_msgs-msg:flag-val is deprecated.  Use pandar_msgs-msg:flag instead.")
  (flag m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <PandarGps>) ostream)
  "Serializes a message object of type '<PandarGps>"
  (cl:let ((__sec (cl:floor (cl:slot-value msg 'stamp)))
        (__nsec (cl:round (cl:* 1e9 (cl:- (cl:slot-value msg 'stamp) (cl:floor (cl:slot-value msg 'stamp)))))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __sec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 0) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __nsec) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __nsec) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'used)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'used)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'used)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'used)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'year)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'year)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'month)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'month)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'day)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'day)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'hour)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'hour)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'minute)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'minute)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'second)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'second)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'fineTime)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'fineTime)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'fineTime)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'fineTime)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'flag)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'flag)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'flag)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'flag)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <PandarGps>) istream)
  "Deserializes a message object of type '<PandarGps>"
    (cl:let ((__sec 0) (__nsec 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __sec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 0) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __nsec) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __nsec) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'stamp) (cl:+ (cl:coerce __sec 'cl:double-float) (cl:/ __nsec 1e9))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'used)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'used)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'used)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'used)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'year)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'year)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'month)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'month)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'day)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'day)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'hour)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'hour)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'minute)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'minute)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'second)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'second)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'fineTime)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'fineTime)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'fineTime)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'fineTime)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'flag)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'flag)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'flag)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'flag)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<PandarGps>)))
  "Returns string type for a message object of type '<PandarGps>"
  "pandar_msgs/PandarGps")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'PandarGps)))
  "Returns string type for a message object of type 'PandarGps"
  "pandar_msgs/PandarGps")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<PandarGps>)))
  "Returns md5sum for a message object of type '<PandarGps>"
  "a075038e2dd27f000f34f827c3e40eec")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'PandarGps)))
  "Returns md5sum for a message object of type 'PandarGps"
  "a075038e2dd27f000f34f827c3e40eec")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<PandarGps>)))
  "Returns full string definition for message of type '<PandarGps>"
  (cl:format cl:nil "time stamp~%uint32 used~%uint16 year~%uint16 month~%uint16 day~%uint16 hour~%uint16 minute~%uint16 second~%uint32 fineTime~%uint32 flag~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'PandarGps)))
  "Returns full string definition for message of type 'PandarGps"
  (cl:format cl:nil "time stamp~%uint32 used~%uint16 year~%uint16 month~%uint16 day~%uint16 hour~%uint16 minute~%uint16 second~%uint32 fineTime~%uint32 flag~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <PandarGps>))
  (cl:+ 0
     8
     4
     2
     2
     2
     2
     2
     2
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <PandarGps>))
  "Converts a ROS message object to a list"
  (cl:list 'PandarGps
    (cl:cons ':stamp (stamp msg))
    (cl:cons ':used (used msg))
    (cl:cons ':year (year msg))
    (cl:cons ':month (month msg))
    (cl:cons ':day (day msg))
    (cl:cons ':hour (hour msg))
    (cl:cons ':minute (minute msg))
    (cl:cons ':second (second msg))
    (cl:cons ':fineTime (fineTime msg))
    (cl:cons ':flag (flag msg))
))
