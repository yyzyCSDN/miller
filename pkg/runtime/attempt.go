// Support for rollbackable trial blocks: the DSL 'attempt { ... } recover(e)
// { ... }' statement and the 'reject "..."' statement.
//
// This file owns:
//
//   - The AttemptReject panic, which unwinds a trial from any depth (reject
//     statements, guarded external effects, and statement runtime errors
//     raised inside called functions). Only the CST attempt node recovers it;
//     any other panic type is an internal program fault and propagates.
//
//   - Per-trial buffering of unredirected stdout/stderr output. Entries are
//     retained in call order and either published when the outermost trial
//     commits, transferred to a parent trial on a nested commit, or discarded
//     on rollback.
//
//   - Deep snapshots of all mutable DSL state (current record, context
//     NR/FNR/FILENAME/FILENUM, out-of-stream variables, local variables and
//     type constraints, regex captures, end-of-stream flag, filter
//     expression) and the hook to rewind the input stream via a
//     TrialController (used by mlr script's next()).

package runtime

import (
	"slices"

	"github.com/johnkerl/miller/v6/pkg/mlrval"
	"github.com/johnkerl/miller/v6/pkg/types"
)

// ----------------------------------------------------------------
// Reject unwinding
// ----------------------------------------------------------------

// AttemptReject is the panic value raised when a trial must be abandoned. It
// is recovered only by the nearest enclosing attempt block.
type AttemptReject struct {
	Reason string
}

func (rej *AttemptReject) Error() string {
	return rej.Reason
}

// RaiseAttemptReject abandons the nearest enclosing attempt block, delivering
// reason to its recover variable. If no attempt block is active, the root
// executor converts the panic into a process-level DSL error.
func RaiseAttemptReject(reason string) {
	panic(&AttemptReject{Reason: reason})
}

// ----------------------------------------------------------------
// Input-stream transaction hook
// ----------------------------------------------------------------

// TrialController is implemented by input streams that support mark/rewind.
// Semantics are strictly nested, matching the lexical nesting of attempt
// blocks:
//
//   - Mark notes the current read position.
//   - Rollback restores the position noted by the matching Mark, so records
//     (and end-of-stream markers) consumed within the trial are replayed on
//     subsequent reads instead of being re-opened or re-read from disk.
//   - Commit releases the matching mark; consumed records become permanent.
//
// A nil TrialController on the State means all three calls are no-ops (e.g.
// mlr put/filter, where next() is unavailable and input position cannot be
// moved by the DSL).
type TrialController interface {
	Mark()
	Rollback()
	Commit()
}

// ----------------------------------------------------------------
// Output buffering
// ----------------------------------------------------------------

// TrialOutputKind selects how a buffered output entry is published.
type TrialOutputKind int

const (
	// TrialOutputStdoutText is unredirected stdout text (print/printn/dump
	// and emit-to-stdout record text when there is no output channel).
	TrialOutputStdoutText TrialOutputKind = iota
	// TrialOutputStderrText is unredirected stderr text (eprint/eprintn/edump).
	TrialOutputStderrText
	// TrialOutputDeferred carries a closure that performs the original write
	// (record-stream append, stdout/stderr handler-manager write, etc.) when
	// the outermost trial commits.
	TrialOutputDeferred
)

// TrialOutput is one buffered piece of output produced inside a trial.
type TrialOutput struct {
	Kind  TrialOutputKind
	Text  string
	Flush func() error
}

// trialFrame collects the outputs produced while one attempt block is active.
type trialFrame struct {
	outputs []TrialOutput
}

// InTrial reports whether execution is currently inside an attempt block.
func (state *State) InTrial() bool {
	return len(state.trialFrames) > 0
}

// PushTrial opens a new trial frame. Called by BeginAttempt.
func (state *State) PushTrial() {
	state.trialFrames = append(state.trialFrames, &trialFrame{})
}

// PopTrial closes the innermost trial frame, returning its buffered outputs.
func (state *State) PopTrial() *trialFrame {
	n := len(state.trialFrames)
	frame := state.trialFrames[n-1]
	state.trialFrames = state.trialFrames[:n-1]
	return frame
}

// BufferTrialOutput appends an output entry to the innermost trial frame. The
// caller must first check InTrial; entries buffered here are invisible until
// the trial commits.
func (state *State) BufferTrialOutput(entry TrialOutput) {
	frame := state.trialFrames[len(state.trialFrames)-1]
	frame.outputs = append(frame.outputs, entry)
}

// AdoptTrialOutputs transfers a nested (successfully committed) trial's
// buffered outputs into its parent frame, preserving call order.
func (state *State) AdoptTrialOutputs(outputs []TrialOutput) {
	frame := state.trialFrames[len(state.trialFrames)-1]
	frame.outputs = append(frame.outputs, outputs...)
}

// ----------------------------------------------------------------
// State snapshots
// ----------------------------------------------------------------

// AttemptSnapshot is the deep-copied DSL state captured on entry to an
// attempt block, restored verbatim when the trial is rejected or fails.
type AttemptSnapshot struct {
	inrec         *mlrval.Mlrmap // nil when no current record
	hasInrec      bool
	context       types.Context
	oosvars       *mlrval.Mlrmap
	stack         *Stack
	regexCaptures [][]string
	atEndOfStream bool
	filterExpr    *mlrval.Mlrval
}

func cloneRegexCaptures(captures [][]string) [][]string {
	out := make([][]string, len(captures))
	for i, frame := range captures {
		out[i] = slices.Clone(frame)
	}
	return out
}

// BeginAttempt captures a deep snapshot of all mutable DSL state, marks the
// input stream position, and opens a trial output frame. The returned
// snapshot is passed back to CommitAttempt or RollbackAttempt exactly once.
func (state *State) BeginAttempt() *AttemptSnapshot {
	snapshot := &AttemptSnapshot{
		context:       *state.Context,
		atEndOfStream: state.AtEndOfStream,
		filterExpr:    state.FilterExpression.Copy(),
		oosvars:       state.Oosvars.Copy(),
		stack:         state.Stack.Snapshot(),
		regexCaptures: cloneRegexCaptures(state.regexCapturesByFrame),
	}
	if state.Inrec != nil {
		snapshot.hasInrec = true
		snapshot.inrec = state.Inrec.Copy()
	}

	if state.TrialController != nil {
		state.TrialController.Mark()
	}

	state.PushTrial()
	return snapshot
}

// restoreSnapshot writes the snapshot's contents back into the live state, in
// place. Pointer-holding fields (Context, Stack) are mutated through their
// existing pointers so long-lived references stay valid.
func (state *State) restoreSnapshot(snapshot *AttemptSnapshot) {
	if state.Context != nil {
		*state.Context = snapshot.context
	} else {
		contextCopy := snapshot.context
		state.Context = &contextCopy
	}
	if snapshot.hasInrec {
		state.Inrec = snapshot.inrec
	} else {
		state.Inrec = nil
	}
	state.Oosvars = snapshot.oosvars
	state.Stack.RestoreFrom(snapshot.stack)
	state.regexCapturesByFrame = cloneRegexCaptures(snapshot.regexCaptures)
	state.AtEndOfStream = snapshot.atEndOfStream
	state.FilterExpression = snapshot.filterExpr
}

// RollbackAttempt abandons a trial: buffered outputs are discarded, all
// mutable DSL state is restored to the snapshot, and the input stream is
// rewound to the snapshot's mark. The recover block runs afterwards at the
// parent trial depth (or with no trial active).
func (state *State) RollbackAttempt(snapshot *AttemptSnapshot) {
	state.PopTrial() // discard this trial's buffered outputs
	state.restoreSnapshot(snapshot)
	if state.TrialController != nil {
		state.TrialController.Rollback()
	}
}

// CommitAttempt makes a successful trial permanent. The input mark is
// released. If a parent trial remains active, the trial's buffered outputs are
// transferred to it; otherwise they are returned for immediate publication, in
// call order.
func (state *State) CommitAttempt(snapshot *AttemptSnapshot) []TrialOutput {
	frame := state.PopTrial()
	if state.TrialController != nil {
		state.TrialController.Commit()
	}
	if state.InTrial() {
		state.AdoptTrialOutputs(frame.outputs)
		return nil
	}
	return frame.outputs
}

// AbandonAttemptWithoutRecover releases a trial whose body died from an
// internal (non-recoverable) fault: the frame and input mark are cleaned up
// before the fault continues to propagate. No recover block runs.
func (state *State) AbandonAttemptWithoutRecover(snapshot *AttemptSnapshot) {
	state.PopTrial()
	if state.TrialController != nil {
		state.TrialController.Rollback()
	}
}
