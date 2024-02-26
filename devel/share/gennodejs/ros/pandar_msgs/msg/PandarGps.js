// Auto-generated. Do not edit!

// (in-package pandar_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class PandarGps {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.stamp = null;
      this.used = null;
      this.year = null;
      this.month = null;
      this.day = null;
      this.hour = null;
      this.minute = null;
      this.second = null;
      this.fineTime = null;
      this.flag = null;
    }
    else {
      if (initObj.hasOwnProperty('stamp')) {
        this.stamp = initObj.stamp
      }
      else {
        this.stamp = {secs: 0, nsecs: 0};
      }
      if (initObj.hasOwnProperty('used')) {
        this.used = initObj.used
      }
      else {
        this.used = 0;
      }
      if (initObj.hasOwnProperty('year')) {
        this.year = initObj.year
      }
      else {
        this.year = 0;
      }
      if (initObj.hasOwnProperty('month')) {
        this.month = initObj.month
      }
      else {
        this.month = 0;
      }
      if (initObj.hasOwnProperty('day')) {
        this.day = initObj.day
      }
      else {
        this.day = 0;
      }
      if (initObj.hasOwnProperty('hour')) {
        this.hour = initObj.hour
      }
      else {
        this.hour = 0;
      }
      if (initObj.hasOwnProperty('minute')) {
        this.minute = initObj.minute
      }
      else {
        this.minute = 0;
      }
      if (initObj.hasOwnProperty('second')) {
        this.second = initObj.second
      }
      else {
        this.second = 0;
      }
      if (initObj.hasOwnProperty('fineTime')) {
        this.fineTime = initObj.fineTime
      }
      else {
        this.fineTime = 0;
      }
      if (initObj.hasOwnProperty('flag')) {
        this.flag = initObj.flag
      }
      else {
        this.flag = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PandarGps
    // Serialize message field [stamp]
    bufferOffset = _serializer.time(obj.stamp, buffer, bufferOffset);
    // Serialize message field [used]
    bufferOffset = _serializer.uint32(obj.used, buffer, bufferOffset);
    // Serialize message field [year]
    bufferOffset = _serializer.uint16(obj.year, buffer, bufferOffset);
    // Serialize message field [month]
    bufferOffset = _serializer.uint16(obj.month, buffer, bufferOffset);
    // Serialize message field [day]
    bufferOffset = _serializer.uint16(obj.day, buffer, bufferOffset);
    // Serialize message field [hour]
    bufferOffset = _serializer.uint16(obj.hour, buffer, bufferOffset);
    // Serialize message field [minute]
    bufferOffset = _serializer.uint16(obj.minute, buffer, bufferOffset);
    // Serialize message field [second]
    bufferOffset = _serializer.uint16(obj.second, buffer, bufferOffset);
    // Serialize message field [fineTime]
    bufferOffset = _serializer.uint32(obj.fineTime, buffer, bufferOffset);
    // Serialize message field [flag]
    bufferOffset = _serializer.uint32(obj.flag, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PandarGps
    let len;
    let data = new PandarGps(null);
    // Deserialize message field [stamp]
    data.stamp = _deserializer.time(buffer, bufferOffset);
    // Deserialize message field [used]
    data.used = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [year]
    data.year = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [month]
    data.month = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [day]
    data.day = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [hour]
    data.hour = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [minute]
    data.minute = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [second]
    data.second = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [fineTime]
    data.fineTime = _deserializer.uint32(buffer, bufferOffset);
    // Deserialize message field [flag]
    data.flag = _deserializer.uint32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a message object
    return 'pandar_msgs/PandarGps';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a075038e2dd27f000f34f827c3e40eec';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    time stamp
    uint32 used
    uint16 year
    uint16 month
    uint16 day
    uint16 hour
    uint16 minute
    uint16 second
    uint32 fineTime
    uint32 flag
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PandarGps(null);
    if (msg.stamp !== undefined) {
      resolved.stamp = msg.stamp;
    }
    else {
      resolved.stamp = {secs: 0, nsecs: 0}
    }

    if (msg.used !== undefined) {
      resolved.used = msg.used;
    }
    else {
      resolved.used = 0
    }

    if (msg.year !== undefined) {
      resolved.year = msg.year;
    }
    else {
      resolved.year = 0
    }

    if (msg.month !== undefined) {
      resolved.month = msg.month;
    }
    else {
      resolved.month = 0
    }

    if (msg.day !== undefined) {
      resolved.day = msg.day;
    }
    else {
      resolved.day = 0
    }

    if (msg.hour !== undefined) {
      resolved.hour = msg.hour;
    }
    else {
      resolved.hour = 0
    }

    if (msg.minute !== undefined) {
      resolved.minute = msg.minute;
    }
    else {
      resolved.minute = 0
    }

    if (msg.second !== undefined) {
      resolved.second = msg.second;
    }
    else {
      resolved.second = 0
    }

    if (msg.fineTime !== undefined) {
      resolved.fineTime = msg.fineTime;
    }
    else {
      resolved.fineTime = 0
    }

    if (msg.flag !== undefined) {
      resolved.flag = msg.flag;
    }
    else {
      resolved.flag = 0
    }

    return resolved;
    }
};

module.exports = PandarGps;
