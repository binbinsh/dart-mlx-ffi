import 'dart:ffi' as ffi;

typedef ArrayCreateOverride =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void> data,
      ffi.Pointer<ffi.Int> shape,
      int dim,
    );

typedef StringCopyOverride = ffi.Pointer<ffi.Char> Function();
typedef StreamCreateOverride = ffi.Pointer<ffi.Void> Function();
typedef StreamGetIndexOverride = int Function(ffi.Pointer<ffi.Void>);
typedef DistributedGroupSplitOverride =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int);
typedef DistributedSendLikeOverride =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      int,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef DistributedRecvOverride =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Int>,
      int,
      int,
      int,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

ArrayCreateOverride? debugArrayFromBoolOverride;
ArrayCreateOverride? debugArrayFromInt32Override;
ArrayCreateOverride? debugArrayFromFloat32Override;
ArrayCreateOverride? debugArrayFromFloat64Override;
ArrayCreateOverride? debugArrayFromInt64Override;
ArrayCreateOverride? debugArrayFromUint64Override;
StringCopyOverride? debugVersionCopyOverride;
void Function(ffi.Pointer<ffi.Char>)? debugDispatchError;
int Function(ffi.Pointer<ffi.Void>)? debugDeviceIsAvailableOverride;
int Function(int)? debugDeviceCountOverride;
StreamCreateOverride? debugStreamNewOverride;
StreamGetIndexOverride? debugStreamGetIndexOverride;
DistributedGroupSplitOverride? debugDistributedGroupSplitOverride;
DistributedSendLikeOverride? debugDistributedSendOverride;
DistributedSendLikeOverride? debugDistributedRecvLikeOverride;
DistributedRecvOverride? debugDistributedRecvOverride;

void resetDebugHooks() {
  debugArrayFromBoolOverride = null;
  debugArrayFromInt32Override = null;
  debugArrayFromFloat32Override = null;
  debugArrayFromFloat64Override = null;
  debugArrayFromInt64Override = null;
  debugArrayFromUint64Override = null;
  debugVersionCopyOverride = null;
  debugDispatchError = null;
  debugDeviceIsAvailableOverride = null;
  debugDeviceCountOverride = null;
  debugStreamNewOverride = null;
  debugStreamGetIndexOverride = null;
  debugDistributedGroupSplitOverride = null;
  debugDistributedSendOverride = null;
  debugDistributedRecvLikeOverride = null;
  debugDistributedRecvOverride = null;
}
